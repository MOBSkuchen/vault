use lld_rx::{link, LldFlavor};

type LinkerResult = Result<(), String>;

// Subsystem for windows builds
pub enum ProdType {
    Console,
    // TODO: Implement Window for windows
    // Window 
}

impl ProdType {
    pub fn to_arg(&self) -> String {
        format!("/subsystem:{}", match self {
            ProdType::Console => "console",
            // ProdType::Window => "window"
        })
    }
}

fn set_entry(lld_flavor: &LldFlavor, args: &mut Vec<String>, entry: String) {
    match lld_flavor {
        LldFlavor::Elf => {
            args.push(format!("-e {}", entry));
        }
        LldFlavor::Wasm => {
            args.push(format!("-e {}", entry));
        }
        LldFlavor::MachO => {
            args.push(format!("-e {}", entry));
        }
        LldFlavor::Coff => {
            args.push(format!("/entry:{}", entry));
        }
    }
}

fn set_output(lld_flavor: &LldFlavor, args: &mut Vec<String>, output: &String) {
    match lld_flavor {
        LldFlavor::Elf => {
            args.push(format!("-o \"{}\"", output));
        }
        LldFlavor::Wasm => {
            args.push(format!("-o \"{}\"", output));
        }
        LldFlavor::MachO => {
            args.push(format!("-o \"{}\"", output));
        }
        LldFlavor::Coff => {
            args.push(format!("/out:{}", output));
        }
    }
}

pub fn lld_link(target: LldFlavor, input_files: Vec<String>, output_path: &String,
            is_lib: bool, mut extra_args: Vec<String>, 
            start_symbol: Option<String>, prod_type: ProdType) -> LinkerResult {
    if is_lib && start_symbol.is_some() {
        println!("Start symbol {} will be discarded as you are building a library.", start_symbol.clone().unwrap());
    }
    
    let mut args: Vec<String> = input_files;
    
    if is_lib {
        args.push("/dll".into())
    }
    
    args.push("/nodefaultlib".to_string());
    
    // Windows only
    if let LldFlavor::Coff = target { args.push(prod_type.to_arg()) }

    if let Some(start_symbol) = start_symbol {
        set_entry(&target, &mut args, start_symbol);
    }
    
    set_output(&target, &mut args, output_path);
    
    args.append(&mut extra_args);
    
    link(target, args).ok()
}