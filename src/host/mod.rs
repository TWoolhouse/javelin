use std::time::Duration;

use crossterm::event::{Event, EventStream, KeyCode};
use futures::StreamExt;
use ratatui::Terminal;
use tokio::{
    select,
    sync::mpsc::{self, UnboundedReceiver, UnboundedSender},
};

use crate::{
    app::App,
    host::action::{Action, Actionable, Packet},
};
pub mod action;

#[derive(Debug)]
pub struct Host {
    app: App,
    exit: bool,

    refresh_rate: Duration,

    tx: UnboundedSender<Action>,
    actions: UnboundedReceiver<Action>,

    events: EventStream,
}

impl Host {
    pub fn new(app: App, refresh_rate: Duration) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            app,
            exit: false,
            refresh_rate,
            events: EventStream::new(),
            actions: rx,
            tx,
        }
    }

    pub async fn run<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        <B as ratatui::backend::Backend>::Error: 'static,
    {
        let mut interval = tokio::time::interval(self.refresh_rate);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        while !self.exit {
            // Render the UI
            terminal.draw(|frame| frame.render_widget(&mut self.app, frame.area()))?;
            // Wait for the next event or action
            select! {
                // Tick the UI at the refresh rate
                _ = interval.tick() => {
                    self.action(Action::Tick);
                }
                // Handle terminal events
                event = self.events.next(), if true => {
                    match event {
                        Some(event) => {
                            self.terminal_event(event?);
                        }
                        None => {},
                    }
                }
                // Handle actions from within the app
                action = self.actions.recv() => {
                    match action {
                        Some(action) => self.action(action),
                        None => {},
                    }
                }
            }

            // Drain all pending actions, so actions raised in the same tick are handled in the same tick.
            while let Some(action) = self.actions.try_recv().ok() {
                self.action(action);
            }
        }

        Ok(())
    }

    fn action(&mut self, action: Action) {
        let packet = Packet {
            action,
            tx: self.tx.clone(),
        };
        self.app
            .action(packet)
            .map(|Packet { action, .. }| match action {
                Action::Quit => self.exit = true,
                _ => {}
            });
    }

    fn terminal_event(&mut self, event: Event) {
        self.app
            .action(event)
            .map(|event| match event.as_key_press_event() {
                Some(key) => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        let _ = self.tx.send(Action::Quit).ok();
                    }
                    _ => {}
                },
                None => {}
            });
    }
}
