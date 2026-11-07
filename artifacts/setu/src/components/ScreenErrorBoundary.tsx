import { Component, type ReactNode } from 'react';
export class ScreenErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (this.state.failed) return <main className="page"><div className="page-inner"><section className="card" role="alert"><div className="empty"><span className="empty-glyph" style={{ color: 'var(--rose)' }} aria-hidden="true">!</span><h1 style={{ fontSize: '1.5rem' }}>This view could not load</h1><p>Please try again. Your session messages are kept in this tab.</p><div className="chip-row" style={{ marginTop: 12 }}><button className="btn btn-primary" onClick={() => window.location.reload()}>Reload view</button><a href="#home" className="btn btn-secondary">Return to overview</a></div></div></section></div></main>;
    return this.props.children;
  }
}
