pub struct CompilationConfig {
    debug: bool,
    pub additional_libs: Vec<String>
}

impl CompilationConfig {
    pub fn new(debug: bool) -> Self {
        Self {debug, additional_libs: Vec::new()}
    }
}

pub enum Directive {
    CompiledOnOs(Vec<String>),
    OnDebug(bool),
    Always,
    Never,
    LinkLib(Vec<String>)
}

impl Directive {
    pub fn handle(self, compilation_config: &mut CompilationConfig) -> bool {
        match self { 
            Directive::CompiledOnOs(systems) => {
                systems.contains(&std::env::consts::OS.to_string())
            }
            Directive::OnDebug(cool) => {
                cool == compilation_config.debug
            }
            Directive::Always => true,
            Directive::LinkLib(mut libs) => { compilation_config.additional_libs.append(&mut libs); true }
            _ => false
        }
    }
}