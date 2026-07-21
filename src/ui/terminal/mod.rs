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
};

// mod waveform;

#[derive(Debug, Default, Clone)]
pub struct State {}

impl Widget for &mut App {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        // self.resolution()
        let fft_pass = &self.pass;
        let bars = fft_pass
            .samples
            .array_windows::<2>()
            .map(|[sample, next]| {
                let freq_lin = crate::app::db::linearise_frequency(sample.freq);
                let next_freq_lin = crate::app::db::linearise_frequency(next.freq);
                Rectangle {
                    x: freq_lin as f64,
                    y: -sample.amp() as f64,
                    width: (next_freq_lin - freq_lin) as f64,
                    height: (sample.amp() * 2.0) as f64,
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>();

        let max_now = (
            Instant::now(),
            fft_pass.peak_sample().amp() as f32,
            fft_pass.samples.iter().map(|s| s.amp()).collect(),
        );

        let max_time = std::time::Duration::from_secs_f32(0.1);

        let pp = self
            .history_db
            .partition_point(|(t, ..)| (max_now.0 - *t) > max_time);
        self.history_db.drain(..pp);
        self.history_db.push_back(max_now);

        // TODO: Add the falling peak that drops over time kinda thing.
        // let maxes = self.history_db.iter()

        let max_height = self
            .history_db
            .iter()
            .map(|(_, h, ..)| *h)
            .reduce(f32::max)
            .unwrap_or(1e-10) as f64;

        let render_min = db_to_linear(-160.0) as f64;

        // HACK: 75% to allow bass to go off screen, to show the rest of the spectrum better, as bass is much louder than the rest of the spectrum.
        let render_bounds = (max_height * 1.0).max(render_min);

        Canvas::default()
            .block(Block::bordered().title("Canvas").title_bottom(vec![
                Span::from(format!("{:.2} Hz ", self.resolution().hertz())),
                Span::from(format!("{} ms ", self.resolution().duration().as_millis())),
                Span::from(format!("{} samples ", self.resolution().samples_out(),)),
                Span::from(format!("{} bars ", bars.len())),
                Span::from(format!("{} Hz ", self.resolution().sample_rate())),
                Span::from(format!("{:.2} dB FS ", linear_to_db(max_height as f32))),
                Span::from(format!(
                    "{:.2} ms ",
                    self.timestamp.duration_since(self.timestamp_last).as_millis()
                )),
            ]))
            .x_bounds([0.0, 1.0])
            .y_bounds([0.0, render_bounds])
            .paint(|ctx| {
                bars.iter().for_each(|bar| {
                    ctx.draw(bar);
                });
            })
            .render(area, buf);

        self.timestamp_last = self.timestamp;
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
