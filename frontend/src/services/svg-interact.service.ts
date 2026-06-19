import * as d3 from 'd3';

const INFO_LINE_SUFFIX = 'info_line';
const TRANSITION_MS = 200;

/**
 * Skip transitions for users who
 * prefer reduced motion.
 */
const transitionDuration = () =>
  window.matchMedia?.(
    '(prefers-reduced-motion: reduce)'
  ).matches
    ? 0
    : TRANSITION_MS;

/**
 * Announces the selected part to
 * assistive technology via a visually
 * hidden, polite live region. The drawn
 * SVG labels are decorative (the info
 * group is `aria-hidden`), so this is
 * what a screen reader actually conveys
 * on selection.
 */
function announce(message: string) {
  let live = d3.select<
    HTMLDivElement, unknown
  >('#interact-live');
  if (live.empty()) {
    live = d3
      .select('body')
      .append('div')
      .attr('id', 'interact-live')
      .attr('role', 'status')
      .attr('aria-live', 'polite')
      .attr('class', 'sr-only');
  }
  live.text(message);
}

const fields = {
  Lights_1: {
    title: 'Light',
    subtitle: '1000W'
  },
  Hood_2: {
    title: 'Hood',
    subtitle: 'Ultimatium Reinforced'
  },
  Tires_3: {
    title: 'Tire',
    subtitle: '200 PSI'
  },
  Windows_4: {
    title: 'Window',
    subtitle: 'Shatterproof'
  },
  door_5: {
    title: 'Door',
    subtitle: 'Easy access'
  }
} as const;

/**
 * Draws info lines for each car part in
 * the SVG. Each line consists of a
 * circle at the center of the part, a
 * vertical line extending upwards, and
 * text labels for the part name and
 * subtitle. The lines are initially
 * hidden and become visible when the
 * user selects on various parts of the 
 * SVG.
 */
export function drawInfoLines() {
  const svg = d3.select('#interactViz');
  // Guard against re-drawing: React
  // StrictMode runs effects twice in
  // dev, which would otherwise
  // duplicate every info line and
  // tab stop.
  if (!svg.select('.' + INFO_LINE_SUFFIX).empty()) {
    return;
  }

  const svgImage = svg.select('image');
  // Dim image to make info lines more visible.
  svgImage.style('filter', 'brightness(0.4)');
  Object.keys(fields).forEach((fieldId) =>
    drawInfoLine(svg, fieldId as keyof typeof fields)
  );
}

/**
 * Draws an info line for a specific
 * car part in the SVG.
 * @param svg The svg selection
 * @param fieldId The field id
 *   corresponding to the class of
 *   the car part in the SVG
 */
function drawInfoLine(
  svg: d3.Selection<
    d3.BaseType,
    unknown,
    HTMLElement,
    undefined
  >,
  fieldId: keyof typeof fields
) {
  svg.selectAll('.' + fieldId).each(function () {
    const bbox = (this as SVGGraphicsElement)
      .getBBox();
    const centerX = bbox.x + bbox.width / 2;
    const centerY = bbox.y + bbox.height / 2;

    const lineGroup = svg
      .append('g')
      .style('pointer-events', 'none')
      .attr('opacity', 0)
      // Decorative: its content is
      // announced via the live region
      // instead.
      .attr('aria-hidden', 'true')
      .attr(
        'class',
        fieldId +
          INFO_LINE_SUFFIX +
          ' ' +
          INFO_LINE_SUFFIX
      );

    const selection = d3.select(this);
    const { title, subtitle } = fields[fieldId];
    const label = `${title}, ${subtitle}`;

    function selectPart() {
      svg.selectAll(".interact_selected")
          .classed("interact_selected", false)
          .transition()
          .duration(transitionDuration())
          .style('opacity', 0)
          .style('cursor', 'pointer');
      svg.selectAll('[role="button"]')
        .attr('aria-pressed', 'false');

      selection.classed("interact_selected", true)
          .attr('aria-pressed', 'true')
          .transition()
          .duration(transitionDuration())
          .style('opacity', 1)
          .style('cursor', 'default');
      lineGroup.classed("interact_selected", true)
          .transition()
          .duration(transitionDuration())
          .style('opacity', 1);

      announce(label);
    }

    selection
      // Expose each part as a focusable,
      // labelled toggle button.
      .attr('role', 'button')
      .attr('tabindex', 0)
      .attr('aria-label', label)
      .attr('aria-pressed', 'false')
      .style('cursor', 'pointer')
      .style('opacity', 0.0)
      // prevents keyboard selection outline
      .on('mousedown',
        (event: MouseEvent) =>
          event.preventDefault()
      )
      .on('click', selectPart)
      .on('keydown', (event: KeyboardEvent) => {
        if (
          event.key === 'Enter' ||
          event.key === ' '
        ) {
          event.preventDefault();
          selectPart();
        }
      });

    lineGroup
      .append('circle')
      .attr('cx', centerX)
      .attr('cy', centerY)
      .attr('r', 2)
      .attr('fill', '#ffffff');

    lineGroup
      .append('rect')
      .attr('x', centerX - .5)
      .attr('y', 40)
      .attr('width', 1)
      .attr('height', centerY - 40)
      .attr('fill', '#ffffff');

    lineGroup
      .append('text')
      .attr('x', centerX)
      .attr('dominant-baseline', 'bottom')
      .attr('text-anchor', 'middle')
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#ffffff')
      .style('font-size', '12px')
      .text(title);

    lineGroup
      .append('text')
      .attr('x', centerX)
      .attr('y', 30)
      .attr('dominant-baseline', 'bottom')
      .attr('text-anchor', 'middle')
      .attr('font-style', 'italic')
      .attr('fill', '#dddddd')
      .style('font-size', '10px')
      .text(subtitle);
  });
}
