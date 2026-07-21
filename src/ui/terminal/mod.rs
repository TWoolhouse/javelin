use std::time::Instant;

use crossterm::event::{Event, KeyCode};
use ratatui::{
    prelude::*,
    widgets::{
        Block,
        canvas::{Canvas, Rectangle},
    },
};
use rodio::math::{db_to_linear, linear_to_db};
use tokio::sync::mpsc::UnboundedSender;

use crate::{
    app::App,
    host::action::{Action, Actionable},
    ui::terminal::waveform::Waveform,
};

// mod control_panel;
mod waveform;

#[derive(Debug, Default, Clone)]
pub struct State {}

impl Widget for &mut App {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        let block = Block::bordered().title("Canvas").title_bottom(vec![
            Span::from(format!("{:.2} Hz ", self.resolution().hertz())),
            Span::from(format!("{} ms ", self.resolution().duration().as_millis())),
            Span::from(format!("{} samples ", self.resolution().samples_out(),)),
            Span::from(format!("{} bars ", self.pass.samples.len())),
            Span::from(format!("{} Hz ", self.resolution().sample_rate())),
            Span::from(format!("{:.2} dB FS ", self.pass.peak.db)),
            Span::from(format!(
                "{:.2} ms ",
                self.pass
                    .instant
                    .device
                    .duration_since(self.pass.instant.callback)
                    .as_millis()
            )),
        ]);

        Waveform {
            pass: &self.pass,
            db_min: -160.0,
        }
        .render(block.inner(area), buf);
        block.render(area, buf);
    }
}

impl Actionable<Event, &UnboundedSender<Action>> for App {
    fn action(&mut self, action: Event, tx: &UnboundedSender<Action>) -> Option<Event> {
        action
            .as_key_press_event()
            .and_then(|ref key| match key.code {
                KeyCode::Up => {
                    let _ = tx.send(Action::FFTSizeUp);
                    None
                }
                KeyCode::Down => {
                    let _ = tx.send(Action::FFTSizeDown);
                    None
                }
                _ => Some(action),
            })
    }
}
