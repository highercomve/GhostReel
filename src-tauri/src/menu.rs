use tauri::{
    Emitter, Manager,
    menu::{Menu, MenuItem, PredefinedMenuItem},
};

const CHECK_FOR_UPDATES: &str = "check-for-updates";

// Type-check the menu code on every desktop target, but install it only on macOS.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn install(app: &tauri::App) -> tauri::Result<()> {
    let menu = Menu::default(app.handle())?;
    let items = menu.items()?;
    let app_menu = items
        .first()
        .and_then(|item| item.as_submenu())
        .ok_or_else(|| std::io::Error::other("Missing macOS application menu"))?;
    let check = MenuItem::with_id(app, CHECK_FOR_UPDATES, "Check for Updates…", true, Some("CmdOrCtrl+U"))?;
    app_menu.insert_items(&[&check, &PredefinedMenuItem::separator(app)?], 2)?;
    app.set_menu(menu)?;
    app.on_menu_event(|app, event| {
        if event.id().as_ref() == CHECK_FOR_UPDATES {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
            if let Err(error) = app.emit_to("main", CHECK_FOR_UPDATES, ()) {
                eprintln!("Failed to request an update check: {error}");
            }
        }
    });
    Ok(())
}
