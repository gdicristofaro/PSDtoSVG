import * as d3 from 'd3';

type BaseSelection = d3.Selection<d3.BaseType, unknown, HTMLElement, unknown>;

const GREEN = 'rgb(19, 104, 14)';
const RED = 'rgb(178, 13, 13)';
const PURPLE = 'rgb(104, 10, 92)';

function animateColor(
  selection: BaseSelection,
  color: string,
  duration: number,
  delay: number
) {
  selection
    .transition()
    .duration(duration)
    .delay(delay)
    .attr('fill', color)
    .attr('stroke', color);
}

function animateIn(
  selection: BaseSelection,
  duration: number,
  delay: number
) {
  selection
    .transition()
    .duration(duration)
    .delay(delay)
    .style('fill-opacity', 0.7)
    .style('stroke-opacity', 1);
}

function vizSetColor(selection: BaseSelection, color: string) {
  selection.attr('fill', color).attr('stroke', color);
}

export function setInitialVizState() {
  const hood = d3.selectAll('#dataviz .Hood_2');
  const lights = d3.selectAll('#dataviz .Lights_1');
  const tires = d3.selectAll('#dataviz .Tires_3');
  const windows = d3.selectAll('#dataviz .Windows_4');
  const door = d3.selectAll('#dataviz .door_5');
  const allpaths = d3.selectAll('#dataviz path');

  allpaths
    .style('fill-opacity', 0)
    .style('stroke-opacity', 0);

  vizSetColor(hood, GREEN);
  vizSetColor(lights, GREEN);
  vizSetColor(tires, GREEN);
  vizSetColor(windows, PURPLE);
  vizSetColor(door, PURPLE);
}

export function runAnimation(
  onAnimationStart: () => void,
  onAnimationComplete: () => void
): void {
  onAnimationStart();

  const hood = d3.selectAll('#dataviz .Hood_2');
  const lights = d3.selectAll('#dataviz .Lights_1');
  const tires = d3.selectAll('#dataviz .Tires_3');
  const windows = d3.selectAll('#dataviz .Windows_4');
  const door = d3.selectAll('#dataviz .door_5');
  const allpaths = d3.selectAll('#dataviz path');

  let startTime = 0;
  animateIn(hood, 2000, startTime);

  startTime += 2000;

  animateIn(lights, 2000, startTime);
  animateIn(tires, 2000, startTime);
  animateColor(hood, RED, 2000, startTime);

  startTime += 2000;

  animateIn(windows, 2000, startTime);
  animateIn(door, 2000, startTime);
  animateColor(hood, PURPLE, 2000, startTime);
  animateColor(lights, PURPLE, 2000, startTime);
  animateColor(tires, PURPLE, 2000, startTime);

  startTime += 2000;

  allpaths
    .transition()
    .duration(2000)
    .delay(startTime)
    .style('fill-opacity', 0)
    .style('stroke-opacity', 0);

  startTime += 2000;

  setTimeout(onAnimationComplete, startTime);
}
