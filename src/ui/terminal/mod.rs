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

use crate::{app::App, audio::fft::FFTResolution, host::action::Actionable};

impl Widget for &mut App {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        let fft_pass = self.fft();
        let bars = fft_pass
            .samples
            .array_windows::<2>()
            .map(|[sample, next]| Rectangle {
                x: sample.freq_lin as f64,
                y: -sample.linear as f64,
                width: (next.freq_lin - sample.freq_lin) as f64,
                height: (sample.linear * 2.0) as f64,
                ..Default::default()
            })
            .collect::<Vec<_>>();

        let max_now = (
            Instant::now(),
            fft_pass.peak_sample().linear as f32,
            fft_pass.samples.into_iter().map(|s| s.linear).collect(),
        );

        let pp = self
            .history_db
            .partition_point(|(t, ..)| (max_now.0 - *t) > std::time::Duration::from_secs_f32(1.0));
        self.history_db.drain(..pp);
        self.history_db.push_back(max_now);

        // let maxes = self.history_db.iter()

        let max_height = self
            .history_db
            .iter()
            .map(|(_, h, ..)| *h)
            .reduce(f32::max)
            .unwrap_or(1e-10) as f64;

        let render_min = db_to_linear(-160.0) as f64;

        Canvas::default()
            .block(Block::bordered().title("Canvas").title_bottom(vec![
                Span::from(format!("{:.2} Hz ", self.resolution().hertz())),
                Span::from(format!("{} ms ", self.resolution().duration().as_millis())),
                Span::from(format!("{} samples ", bars.len())),
                Span::from(format!("{} Hz ", self.resolution().sample_rate())),
                Span::from(format!("{:.2} dB FS ", linear_to_db(max_height as f32))),
                Span::from(format!(
                    "{:.2} ms ",
                    self.timestamp.duration_since(self.timestamp_last).as_millis()
                )),
            ]))
            .x_bounds([0.0, 1.0])
            .y_bounds([-max_height.max(render_min), max_height.max(render_min)])
            .paint(|ctx| {
                bars.iter().for_each(|bar| {
                    ctx.draw(bar);
                });
            })
            .render(area, buf);

        self.timestamp_last = self.timestamp;
    }
}

impl Actionable<Event> for App {
    fn action(&mut self, action: Event) -> Option<Event> {
        action
            .as_key_press_event()
            .and_then(|ref key| match key.code {
                KeyCode::Up => {
                    self.resize(FFTResolution::from_samples(
                        self.resolution().samples() + 1,
                        self.resolution().sample_rate(),
                    ));
                    None
                }
                KeyCode::Down => {
                    // TODO: Make this go down a step, instead of half, search for the next optimal below.
                    // FIXME: Prevent 0 samples
                    self.resize(FFTResolution::from_samples(
                        self.resolution().samples() / 2,
                        self.resolution().sample_rate(),
                    ));
                    None
                }
                _ => Some(action),
            })
    }
}
