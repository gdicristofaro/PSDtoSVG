import React from 'react';

const InteractCodeSnippet: React.FC = () => (
  <pre className="w-0">
    {`
import * as d3 from 'd3';

const INFO_LINE_SUFFIX = 'info_line';

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
 * Draws info lines for each car part in the SVG. Each line consists of a circle
 * at the center of the part, a vertical line extending upwards, and text labels
 * for the part name and subtitle. The lines are initially hidden and become 
 * visible when the user hovers over the corresponding part in the SVG. The
 * background image is dimmed to make the info lines more visible.
 */
export function drawInfoLines() {
  const svg = d3.select('#interactViz');
  const svgImage = svg.select('image');
  svgImage.style('filter', 'brightness(0.4)'); // Dim the image to make info lines more visible
  Object.keys(fields).forEach((fieldId) =>
    drawInfoLine(svg, fieldId as keyof typeof fields)
  );
}

/**
 * Draws an info line for a specific car part in the SVG.
 * @param svg THe svg selection
 * @param fieldId The field id corresponding to the class of the car part in the SVG
 */
function drawInfoLine(
  svg: d3.Selection<d3.BaseType, unknown, HTMLElement, undefined>,
  fieldId: keyof typeof fields
) {
  svg.selectAll('.' + fieldId).each(function () {
    const bbox = (this as SVGGraphicsElement).getBBox();
    const centerX = bbox.x + bbox.width / 2;
    const centerY = bbox.y + bbox.height / 2;

    const lineGroup = svg
      .append('g')
      .style('pointer-events', 'none')
      .attr('opacity', 0)
      .attr('class', fieldId + INFO_LINE_SUFFIX + " " + INFO_LINE_SUFFIX);

    const selection = d3.select(this);

    selection
      .style('opacity', 0.0)
      .on('click', function () {
        svg.selectAll(".interact_selected")
            .classed("interact_selected", false)
            .transition()
            .duration(200)
            .style('opacity', 0);

        selection.classed("interact_selected", true)
            .transition()
            .duration(200)
            .style('opacity', 1);
        lineGroup.classed("interact_selected", true)
            .transition()
            .duration(200)
            .style('opacity', 1);
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
      .text(fields[fieldId].title);

    lineGroup
      .append('text')
      .attr('x', centerX)
      .attr('y', 30)
      .attr('dominant-baseline', 'bottom')
      .attr('text-anchor', 'middle')
      .attr('font-style', 'italic')
      .attr('fill', '#dddddd')
      .style('font-size', '10px')
      .text(fields[fieldId].subtitle);
  });
}
`}
  </pre>
);

export default InteractCodeSnippet;
