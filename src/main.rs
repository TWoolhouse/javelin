use display_info::DisplayInfo;

use crate::{app::App, host::Host};

mod app;
mod audio;
mod host;
mod ui;
// mod controller;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut terminal = ratatui::init();

    let app = App::default();

    let refresh_rate = DisplayInfo::all()
        .ok()
        .and_then(|displays| displays.into_iter().map(|d| d.frequency).reduce(f32::max))
        .unwrap_or(60.0);
    let mut host = Host::new(
        app,
        std::time::Duration::from_millis((1000.0 / refresh_rate) as u64),
    );

    let result = {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(host.run(&mut terminal))
    };

    ratatui::restore();
    result
}
