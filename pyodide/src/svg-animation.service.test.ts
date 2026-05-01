import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const transitionMock = {
  duration: vi.fn(() => transitionMock),
  delay: vi.fn(() => transitionMock),
  attr: vi.fn(() => transitionMock),
  style: vi.fn(() => transitionMock)
} as { duration: () => unknown; delay: () => unknown; attr: () => unknown; style: () => unknown };

const selectionMock = {
  transition: vi.fn(() => transitionMock),
  attr: vi.fn(() => selectionMock),
  style: vi.fn(() => selectionMock)
} as { transition: () => unknown; attr: () => unknown; style: () => unknown };

vi.mock('d3', () => ({
  selectAll: vi.fn(() => selectionMock)
}));

describe('svg-animation.service', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('starts and completes the animation with the expected callback calls', async () => {
    const { runAnimation } = await import('./svg-animation.service');
    const onAnimationStart = vi.fn();
    const onAnimationComplete = vi.fn();

    runAnimation(onAnimationStart, onAnimationComplete);

    expect(onAnimationStart).toHaveBeenCalledTimes(1);
    expect(onAnimationComplete).not.toHaveBeenCalled();

    vi.runAllTimers();

    expect(onAnimationComplete).toHaveBeenCalledTimes(1);
  });

  it('uses d3.selectAll and transition on the expected selectors', async () => {
    const { runAnimation } = await import('./svg-animation.service');
    const onAnimationStart = vi.fn();
    const onAnimationComplete = vi.fn();

    runAnimation(onAnimationStart, onAnimationComplete);

    const { selectAll } = await import('d3');
    expect(selectAll).toHaveBeenCalledWith('#dataviz .Hood_3');
    expect(selectAll).toHaveBeenCalledWith('#dataviz .Lights_4');
    expect(selectAll).toHaveBeenCalledWith('#dataviz .Tires_2');
    expect(selectAll).toHaveBeenCalledWith('#dataviz .Windows_1');
    expect(selectAll).toHaveBeenCalledWith('#dataviz .door_0');
    expect(selectAll).toHaveBeenCalledWith('#dataviz path');

    expect(selectionMock.attr).toHaveBeenCalled();
    expect(selectionMock.transition).toHaveBeenCalled();
  });
});
