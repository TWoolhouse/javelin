use tokio::sync::mpsc;

pub trait Actionable<A, C> {
    fn action(&mut self, action: A, context: C) -> Option<A>;
}

impl<T, A, C> Actionable<A, C> for Option<T>
where
    T: Actionable<A, C>,
{
    fn action(&mut self, action: A, context: C) -> Option<A> {
        match self {
            Some(on) => on.action(action, context),
            None => Some(action),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Tick,
    Render,
    Quit,
    FFTSizeUp,
    FFTSizeDown,
}
