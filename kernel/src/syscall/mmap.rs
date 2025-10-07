// SPDX-License-Identifier: MPL-2.0

//! Implements the `mmap` syscall for a kernel. This allows user processes
//! to map files or anonymous memory into their virtual address space.

use align_ext::AlignExt;
use aster_rights::Rights;

use super::SyscallReturn;
use crate::{
    fs::file_table::{get_file_fast, FileDesc},
    prelude::*,
    vm::{perms::VmPerms, vmar::is_userspace_vaddr, vmo::VmoOptions},
};

/// Entry point for the mmap system call.
/// Parameters correspond to the standard Linux `mmap` syscall:
/// - `addr`: Suggested start address of the mapping.
/// - `len`: Length of the mapping.
/// - `perms`: Protection bits (PROT_READ/WRITE/EXEC).
/// - `flags`: Mapping flags (MAP_SHARED, MAP_ANONYMOUS, etc.).
/// - `fd`: File descriptor (if mapping a file).
/// - `offset`: Offset in the file to map from.
/// - `ctx`: Process/thread context.
pub fn sys_mmap(
    addr: u64,
    len: u64,
    perms: u64,
    flags: u64,
    fd: u64,
    offset: u64,
    ctx: &Context,
) -> Result<SyscallReturn> {
    // Convert numeric flags/perms into typed enums/bitflags.
    let perms = VmPerms::from_bits_truncate(perms as u32);
    let option = MMapOptions::try_from(flags as u32)?;
    
    // Delegate actual logic to `do_sys_mmap`.
    let res = do_sys_mmap(
        addr as usize,
        len as usize,
        perms,
        option,
        fd as _,
        offset as usize,
        ctx,
    )?;
    Ok(SyscallReturn::Return(res as _))
}

/// The core logic of the mmap syscall.
/// This function validates parameters, prepares a mapping, and
/// creates a `VMO` (virtual memory object) backed by either
/// anonymous memory or a file.
fn do_sys_mmap(
    addr: Vaddr,
    len: usize,
    vm_perms: VmPerms,
    mut option: MMapOptions,
    fd: FileDesc,
    offset: usize,
    ctx: &Context,
) -> Result<Vaddr> {
    debug!(
        "addr = 0x{:x}, len = 0x{:x}, perms = {:?}, option = {:?}, fd = {}, offset = 0x{:x}",
        addr, len, vm_perms, option, fd, offset
    );

    // MAP_FIXED_NOREPLACE implies MAP_FIXED (per POSIX)
    if option.flags.contains(MMapFlags::MAP_FIXED_NOREPLACE) {
        option.flags.insert(MMapFlags::MAP_FIXED);
    }

    // Basic validation of flags and address.
    check_option(addr, len, &option)?;

    // Validate length and prevent overflow issues.
    if len == 0 {
        return_errno_with_message!(Errno::EINVAL, "mmap len cannot be zero");
    }
    if len > isize::MAX as usize {
        return_errno_with_message!(Errno::ENOMEM, "mmap len too large");
    }

    // Round length up to page size.
    let len = len.align_up(PAGE_SIZE);

    // Offsets must be page-aligned.
    if offset % PAGE_SIZE != 0 {
        return_errno_with_message!(Errno::EINVAL, "mmap only support page-aligned offset");
    }
    // Prevent integer overflow on offset + len.
    offset.checked_add(len).ok_or(Error::with_message(
        Errno::EOVERFLOW,
        "integer overflow when (offset + len)",
    ))?;
    // Prevent addr + len overflow.
    if addr > isize::MAX as usize - len {
        return_errno_with_message!(Errno::ENOMEM, "mmap (addr + len) too large");
    }

    // On x86, `PROT_WRITE` implies `PROT_READ`.
    #[cfg(target_arch = "x86_64")]
    let vm_perms = if !vm_perms.contains(VmPerms::READ) && vm_perms.contains(VmPerms::WRITE) {
        vm_perms | VmPerms::READ
    } else {
        vm_perms
    };

    let mut vm_may_perms = VmPerms::ALL_MAY_PERMS;

    // Retrieve the current process's root VMAR (Virtual Memory Address Region).
    let user_space = ctx.user_space();
    let root_vmar = user_space.root_vmar();

    // Build a new mapping options object.
    let vm_map_options = {
        let mut options = root_vmar.new_map(len, vm_perms)?;
        let flags = option.flags;

        // MAP_FIXED: must map exactly at `addr`
        if flags.contains(MMapFlags::MAP_FIXED) {
            options = options.offset(addr).can_overwrite(true);
        } else if flags.contains(MMapFlags::MAP_32BIT) {
            // TODO: enforce <2GB mapping range for MAP_32BIT
            warn!("MAP_32BIT is not supported");
        }

        // Shared mappings must be marked as such.
        if option.typ() == MMapType::Shared {
            options = options.is_shared(true);
        }

        // Handle anonymous mappings.
        if option.flags.contains(MMapFlags::MAP_ANONYMOUS) {
            if offset != 0 {
                return_errno_with_message!(
                    Errno::EINVAL,
                    "offset must be zero for anonymous mapping"
                );
            }

            // Shared anonymous mapping: share the same backing VMO.
            if option.typ() == MMapType::Shared {
                let shared_vmo = {
                    let vmo_options: VmoOptions<Rights> = VmoOptions::new(len);
                    vmo_options.alloc()?
                };
                options = options.vmo(shared_vmo);
            }
        } else {
            // File-backed mapping: get the file and verify permissions.
            let mut file_table = ctx.thread_local.borrow_file_table_mut();
            let file = get_file_fast!(&mut file_table, fd);

            let access_mode = file.access_mode();
            if vm_perms.contains(VmPerms::READ) && !access_mode.is_readable() {
                return_errno!(Errno::EACCES);
            }
            if option.typ() == MMapType::Shared && !access_mode.is_writable() {
                if vm_perms.contains(VmPerms::WRITE) {
                    return_errno!(Errno::EACCES);
                }
                vm_may_perms.remove(VmPerms::MAY_WRITE);
            }

            // Build mapping options with file-backed VMO.
            options = options
                .may_perms(vm_may_perms)
                .mappable(file.mappable()?) // source: file's memory object
                .vmo_offset(offset)
                .handle_page_faults_around();
        }

        options
    };

    // Finally, create the mapping in the virtual address space.
    let map_addr = vm_map_options.build()?;

    Ok(map_addr)
}

/// Basic validation for mapping options before creating a VMO.
fn check_option(addr: Vaddr, size: usize, option: &MMapOptions) -> Result<()> {
    // Type must not be `File` (invalid)
    if option.typ() == MMapType::File {
        return_errno_with_message!(Errno::EINVAL, "Invalid mmap type");
    }

    let map_end = addr.checked_add(size).ok_or(Errno::EINVAL)?;
    // If MAP_FIXED is used, ensure address range is within user space.
    if option.flags().contains(MMapFlags::MAP_FIXED)
        && !(is_userspace_vaddr(addr) && is_userspace_vaddr(map_end - 1))
    {
        return_errno_with_message!(Errno::EINVAL, "Invalid mmap fixed addr");
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// mmap flag definitions (mostly mirror Linux semantics)
// -----------------------------------------------------------------------------

// Low 4 bits encode the map type (shared/private/etc.)
const MAP_TYPE: u32 = 0xf;

#[derive(Copy, Clone, PartialEq, Debug, TryFromInt)]
#[repr(u8)]
pub enum MMapType {
    File = 0x0, // Invalid
    Shared = 0x1,
    Private = 0x2,
    SharedValidate = 0x3,
}

bitflags! {
    pub struct MMapFlags : u32 {
        const MAP_FIXED           = 0x10;
        const MAP_ANONYMOUS       = 0x20;
        const MAP_32BIT           = 0x40;
        const MAP_GROWSDOWN       = 0x100;
        const MAP_DENYWRITE       = 0x800;
        const MAP_EXECUTABLE      = 0x1000;
        const MAP_LOCKED          = 0x2000;
        const MAP_NORESERVE       = 0x4000;
        const MAP_POPULATE        = 0x8000;
        const MAP_NONBLOCK        = 0x10000;
        const MAP_STACK           = 0x20000;
        const MAP_HUGETLB         = 0x40000;
        const MAP_SYNC            = 0x80000;
        const MAP_FIXED_NOREPLACE = 0x100000;
    }
}

/// Struct representing parsed mmap options.
#[derive(Debug)]
pub struct MMapOptions {
    typ: MMapType,
    flags: MMapFlags,
}

impl TryFrom<u32> for MMapOptions {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        let typ_raw = (value & MAP_TYPE) as u8;
        let typ = MMapType::try_from(typ_raw)?;

        let flags_raw = value & !MAP_TYPE;
        let Some(flags) = MMapFlags::from_bits(flags_raw) else {
            return Err(Error::with_message(Errno::EINVAL, "unknown mmap flags"));
        };
        Ok(MMapOptions { typ, flags })
    }
}

impl MMapOptions {
    pub fn typ(&self) -> MMapType {
        self.typ
    }

    pub fn flags(&self) -> MMapFlags {
        self.flags
    }
}