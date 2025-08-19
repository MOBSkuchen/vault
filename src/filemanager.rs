use crate::comp_errors::{CompResult, CompilerError};
use crate::lexer::CodePosition;
use annotate_snippets::Snippet;
use std::path::{Path, PathBuf};
use std::{fs, io};
use crate::utils::relative_path;

fn resolve_path<P: AsRef<Path>>(input_path: P) -> io::Result<PathBuf> {
    let path = input_path.as_ref();
    let absolute_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    if absolute_path.exists() || absolute_path.parent().map(|p| p.exists()).unwrap_or(false) {
        Ok(absolute_path)
    } else {
        Err(io::Error::new(io::ErrorKind::NotFound, "Invalid path"))
    }
}

#[derive(Debug)]
pub struct FileManager {
    pub input_file: String,
    pub file_path: PathBuf,
    content: String,
}

impl FileManager {
    pub fn new(file_path: PathBuf, input_file: String) -> CompResult<Self> {
        if !file_path.exists() {
            Err(CompilerError::FileNotAccessible(
                input_file,
                !file_path.parent().is_some_and(|t| t.exists()),
            ))
        } else {
            let content = fs::read_to_string(&file_path);
            if let Ok(content) = content {
                Ok(Self {
                    input_file,
                    file_path,
                    content,
                })
            } else {
                Err(CompilerError::FileCorrupted(input_file))
            }
        }
    }

    pub fn new_from(file: String) -> CompResult<Self> {
        let x = resolve_path(&file);
        if let Ok(m) = x {
            Self::new(m, file)
        } else {
            Err(CompilerError::FileNotAccessible(file, true))
        }
    }

    pub fn get_content(&self) -> String {
        self.content.clone()
    }

    pub fn get_surrounding_slice(&self, line_index: usize) -> (String, usize) {
        let lines: Vec<&str> = self.content.lines().collect();
        let total_lines = lines.len();

        if total_lines == 0 {
            return (String::new(), 0);
        }

        let mut snippet = String::new();
        let mut offset = 0;

        if line_index > 0 {
            snippet.push_str(lines[line_index - 1]);
            snippet.push('\n');
            offset += lines[line_index - 1].len() + 1;
        }

        if line_index < total_lines {
            snippet.push_str(lines[line_index]);
            snippet.push('\n');
        }

        if line_index + 1 < total_lines {
            snippet.push_str(lines[line_index + 1]);
            snippet.push('\n');
        }

        (snippet, offset)
    }

    pub fn get_code_snippet(&self, code_position: &CodePosition) -> (Snippet, usize) {
        // TODO: Remove this super evil magic trick
        let sor_slc = self.get_surrounding_slice(code_position.line_start);
        let clean_path = &self.input_file;
        (
            Snippet::source(sor_slc.0.leak())
                .line_start(if code_position.line_start == 0 {
                    code_position.line_start + 1
                } else {
                    code_position.line_start
                })
                .origin(relative_path(clean_path).to_string().leak()),
            sor_slc.1
        )
    }
}
