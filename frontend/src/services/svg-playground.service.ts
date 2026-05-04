export interface PlaygroundState {
  dashArray: 'none' | 'small' | 'medium' | 'large';
  scale: number;
  fillOpacity: number;
  strokeOpacity: number;
  strokeWidth: number;
  doorColor: number;
  tireColor: number;
  hoodColor: number;
  lightsColor: number;
  windowColor: number;
}

export const initialPlaygroundState: PlaygroundState = {
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
};

export function getBounded(num: number) {
  return Math.min(1, Math.max(0, num));
}

export function getScale(num: number) {
  const perc = 1 + getBounded(num) * 0.5;
  const xoff = (-800 * (perc - 1)) / 2;
  const yoff = (-561 * (perc - 1)) / 2;

  return `translate(${Math.floor(xoff)},${Math.floor(yoff)}) scale(${perc})`;
}

export function getBlendedColor(
  r1: number,
  g1: number,
  b1: number,
  r2: number,
  g2: number,
  b2: number,
  num: number
) {
  const r = Math.floor(r2 * num + r1 * (1 - num));
  const g = Math.floor(g2 * num + g1 * (1 - num));
  const b = Math.floor(b2 * num + b1 * (1 - num));
  return `rgb(${r},${g},${b})`;
}

export function getColor(num: number) {
  num = getBounded(num);

  if (num < 1 / 3) {
    return getBlendedColor(255, 0, 0, 0, 255, 0, num * 3);
  } else if (num < 2 / 3) {
    return getBlendedColor(0, 255, 0, 0, 0, 255, (num - 1 / 3) * 3);
  }

  return getBlendedColor(0, 0, 255, 255, 0, 0, (num - 2 / 3) * 3);
}

export function getDash(dash: PlaygroundState['dashArray']) {
  switch (dash) {
    case 'small':
      return '2';
    case 'medium':
      return '10,5';
    case 'large':
      return '20,20';
    case 'none':
    default:
      return 'none';
  }
}

export function updatePlaygroundState(
  prev: PlaygroundState,
  field: keyof PlaygroundState,
  value: string | number
): PlaygroundState {
  return {
    ...prev,
    [field]:
      typeof value === 'string' && !isNaN(Number(value)) && field !== 'dashArray'
        ? Number(value)
        : value
  };
}
