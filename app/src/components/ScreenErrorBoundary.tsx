import { Component, type ReactNode } from 'react';
export class ScreenErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (this.state.failed) return <main><div className="mx-auto px-4 sm:px-6 lg:px-8"><section className="content-card empty-state" role="alert"><h1 className="text-headline-lg font-semibold">This view could not load</h1><p className="my-4">Please try again. Your session messages are kept in this tab.</p><button className="primary-action" onClick={() => window.location.reload()}>Reload view</button><a href="#home" className="secondary-action ml-3">Return to overview</a></section></div></main>;
    return this.props.children;
  }
}
