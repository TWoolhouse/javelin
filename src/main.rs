use display_info::DisplayInfo;

use crate::{app::App, host::Host};

mod app;
mod audio;
mod host;
mod ui;
// mod controller;
#[cfg(feature = "server")]
mod server;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    logging()?;

    let app = App::default();

    let refresh_rate = DisplayInfo::all()
        .ok()
        .and_then(|displays| displays.into_iter().map(|d| d.frequency).reduce(f32::max))
        .unwrap_or(60.0);
    let mut host = Host::new(
        app,
        std::time::Duration::from_millis((1000.0 / refresh_rate) as u64),
    );

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let rocket_handle = {
                #[cfg(feature = "server")]
                {
                    let rocket = server::build().await?;
                    let shutdown = rocket.shutdown();

                    let rocket_handle = tokio::spawn(rocket.launch());
                    async move || {
                        shutdown.notify();
                        rocket_handle.await.map_err(|e| {
                            let b: Box<dyn std::error::Error> = Box::new(e);
                            b
                        })?;
                        Ok(())
                    }
                }
                #[cfg(not(feature = "server"))]
                {
                    async move || Ok(())
                }
            };

            let mut terminal = ratatui::init();
            let host_res = host.run(&mut terminal).await;
            ratatui::restore();

            match (host_res, rocket_handle().await) {
                (Ok(_), Ok(_)) => Ok(()),
                (Err(e), _) => Err(e),
                (_, Err(e)) => Err(e),
            }
        })
}

fn logging() -> Result<(), fern::InitError> {
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{}][{}] {}",
                record.level(),
                record.target(),
                message
            ))
        })
        .level(log::LevelFilter::Debug)
        .chain(
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open("javelin.log")?,
        )
        .apply()?;
    Ok(())
}
