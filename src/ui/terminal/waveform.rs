use ratatui::{
    prelude::*,
    widgets::canvas::{Canvas, Rectangle},
};
use rodio::math::db_to_linear;

use crate::audio::pipeline::pass::Pass;

#[derive(Debug)]
pub struct Waveform<'p> {
    pub pass: &'p Pass,
    pub db_min: f32,
}

impl<'p> Widget for &Waveform<'p> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let bars = self
            .pass
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

        let max_height = self.pass.peak.amp() as f64;
        let render_min = db_to_linear(self.db_min) as f64;
        let render_bounds = (max_height * 1.0).max(render_min);

        Canvas::default()
            .x_bounds([0.0, 1.0])
            .y_bounds([0.0, render_bounds])
            .paint(|ctx| {
                bars.iter().for_each(|bar| {
                    ctx.draw(bar);
                });
            })
            .render(area, buf);
    }
}
