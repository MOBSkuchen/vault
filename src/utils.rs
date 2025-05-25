use std::collections::HashSet;
use std::hash::Hash;
use inkwell::context::Context;
use inkwell::llvm_sys::prelude::LLVMModuleRef;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::Module;

#[allow(clippy::needless_range_loop)]
pub fn dedup<T: Hash + Eq>(v: &mut Vec<T>) {
    let mut set = HashSet::new();
    let mut indices = Vec::new();

    for i in 0..v.len() {
        if !set.insert(&v[i]){
            indices.push(i);
        }
    }

    for (pos, e) in indices.iter().enumerate() {
        v.remove(*e - pos);
    }
}

pub fn relative_path(p: &str) -> &str {
    // This *should* always work if compiler is accessing the nested files
    // Otherwise, we will return the full path
    p.strip_prefix(
        &std::env::current_dir()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string(),
    )
        .or(Some(p))
        .expect("There is no reason")
}

#[macro_export]
macro_rules! double_unwrap {
    ($expr:expr) => {{
        match $expr {
            Ok(inner) => match inner {
                Ok(value) => Ok(Ok(value)),
                Err(e) => Ok(Err(e)),
            },
            Err(e) => Err(e),
        }
    }};
}