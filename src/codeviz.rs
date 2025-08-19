use crate::comp_errors::{CodeError, CodeWarning};
use crate::filemanager::FileManager;
use annotate_snippets::{Level, Renderer, Snippet};
use crate::lexer::CodePosition;

fn make_alive_snippet(snip: Snippet, offset: usize, position: CodePosition, pointer: Option<String>) -> Snippet {
    snip.annotation(match pointer {
        None => Level::Error.span(position.range(offset)),
        Some(_) => Level::Error
            .span(position.range(offset))
            .label(pointer.unwrap().leak()),
    })
}

pub fn print_code_error(code_error: CodeError, file_manager: &FileManager) {
    let (mut snip, offset) = file_manager.get_code_snippet(&code_error.position);
    snip = make_alive_snippet(snip, offset, code_error.position, code_error.pointer);

    let mut snippets = vec![snip];

    for ext in code_error.exts {
        let (mut snip, offset) = file_manager.get_code_snippet(&ext.position);
        snip = snip.annotation(Level::Error.span(ext.position.range(offset)).label(ext.pointer.leak()));
        snippets.push(snip)
    }

    let mut footers = vec![Level::Error.title(code_error.footer.as_str())];

    for note in &code_error.notes {
        footers.push(Level::Note.title(note))
    }

    let id_fmt = format!("{:#04x}", code_error.code_error_type as usize);
    let msg = Level::Error
        .title(code_error.title.as_str())
        .id(&id_fmt)
        .snippets(snippets)
        .footers(footers);

    let renderer = Renderer::styled();
    anstream::println!("{}", renderer.render(msg));
}

pub fn print_code_warn(code_warn: CodeWarning, file_manager: &FileManager) {
    let (mut snip, offset) = file_manager.get_code_snippet(&code_warn.position);
    snip = make_alive_snippet(snip, offset, code_warn.position, code_warn.pointer);

    let mut footers = vec![Level::Warning.title(code_warn.footer.as_str())];

    for note in &code_warn.notes {
        footers.push(Level::Note.title(note))
    }

    let id_fmt = format!("{:#04x}", code_warn.code_warn_type as usize);
    let msg = Level::Warning
        .title(code_warn.title.as_str())
        .id(&id_fmt)
        .snippet(snip)
        .footers(footers);

    let renderer = Renderer::styled();
    anstream::println!("{}", renderer.render(msg));
}
