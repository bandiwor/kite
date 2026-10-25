//! The standard library part that is written in Kite (`prelude.kite`): `Option`, `Result`
//! and `Error`.  The VM keeps the three types so natives and `?` can work with them.

use crate::vm::Vm;

pub fn install(vm: &mut Vm) {
    if let Err(e) = vm.eval("<prelude>", include_str!("prelude.kite")) {
        panic!("the prelude failed to load:\n{}", e);
    }
    for (name, slot) in [("Option", 0), ("Result", 1), ("Error", 2)] {
        let ty = vm.get_global(name).expect("prelude type");
        match slot {
            0 => vm.option_ty = ty,
            1 => vm.result_ty = ty,
            _ => vm.error_ty = ty,
        }
    }
}
