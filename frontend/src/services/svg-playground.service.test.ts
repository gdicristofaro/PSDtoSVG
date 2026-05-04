import { describe, expect, it } from 'vitest';
import {
  getColor,
  getDash,
  getScale,
  initialPlaygroundState,
  updatePlaygroundState
} from './svg-playground.service';

describe('svg-playground.service', () => {
  it('provides the expected initial playground state', () => {
    expect(initialPlaygroundState).toEqual({
      dashArray: 'none',
      scale: 0,
      fillOpacity: 50,
      strokeOpacity: 80,
      strokeWidth: 5,
      doorColor: 1,
      tireColor: 0.8,
      hoodColor: 0.6,
      lightsColor: 0.4,
      windowColor: 0.15
    });
  });

  it('returns the correct transform string for scale', () => {
    expect(getScale(0)).toBe('translate(0,0) scale(1)');
    expect(getScale(1)).toBe('translate(-200,-141) scale(1.5)');
  });

  it('returns a blended color for intermediate values', () => {
    expect(getColor(0)).toBe('rgb(255,0,0)');
    expect(getColor(0.5)).toBe('rgb(0,127,127)');
    expect(getColor(0.99)).toBe('rgb(247,0,7)');
  });

  it('returns the correct dash strings', () => {
    expect(getDash('none')).toBe('none');
    expect(getDash('small')).toBe('2');
    expect(getDash('medium')).toBe('10,5');
    expect(getDash('large')).toBe('20,20');
  });

  it('updates numeric state values from string input', () => {
    const prev = initialPlaygroundState;
    const next = updatePlaygroundState(prev, 'fillOpacity', '75');
    expect(next.fillOpacity).toBe(75);
    expect(typeof next.fillOpacity).toBe('number');
  });

  it('keeps dashArray values as strings', () => {
    const prev = initialPlaygroundState;
    const next = updatePlaygroundState(prev, 'dashArray', 'medium');
    expect(next.dashArray).toBe('medium');
  });
});
