use crate::comp_errors::{CodeError, CodeWarning};
use crate::filemanager::FileManager;
use annotate_snippets::{Level, Renderer};

pub fn print_code_error(code_error: CodeError, file_manager: &FileManager) {
    let (mut snip, offset) = file_manager.get_code_snippet(&code_error.position);
    snip = snip.annotation(match code_error.pointer {
        None => Level::Error.span(code_error.position.range(offset)),
        Some(_) => Level::Error
            .span(code_error.position.range(offset))
            .label(code_error.pointer.unwrap().leak()),
    });

    let mut footers = vec![Level::Error.title(code_error.footer.as_str())];

    for note in &code_error.notes {
        footers.push(Level::Note.title(note))
    }

    let id_fmt = format!("{:#04x}", code_error.code_error_type as usize);
    let msg = Level::Error
        .title(code_error.title.as_str())
        .id(&*id_fmt)
        .snippet(snip)
        .footers(footers);

    let renderer = Renderer::styled();
    anstream::println!("{}", renderer.render(msg));
}

pub fn print_code_warn(code_warn: CodeWarning, file_manager: &FileManager) {
    let (mut snip, offset) = file_manager.get_code_snippet(&code_warn.position);
    snip = snip.annotation(match code_warn.pointer {
        None => Level::Warning.span(code_warn.position.range(offset)),
        Some(_) => Level::Warning
            .span(code_warn.position.range(offset))
            .label(code_warn.pointer.unwrap().leak()),
    });

    let mut footers = vec![Level::Warning.title(code_warn.footer.as_str())];

    for note in &code_warn.notes {
        footers.push(Level::Note.title(note))
    }

    let id_fmt = format!("{:#04x}", code_warn.code_warn_type as usize);
    let msg = Level::Warning
        .title(code_warn.title.as_str())
        .id(&*id_fmt)
        .snippet(snip)
        .footers(footers);

    let renderer = Renderer::styled();
    anstream::println!("{}", renderer.render(msg));
}
