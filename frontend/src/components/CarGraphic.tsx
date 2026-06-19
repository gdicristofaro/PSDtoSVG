import React, { useRef, useEffect } from 'react';
import carGraphicRaw from './CarGraphic.svg?raw';

const svgBase = carGraphicRaw.replace(/^<\?xml[^?]*\?>\s*/s, '');

const CarGraphic: React.FC<{ id: string; label: string; interactive?: boolean }> = ({
  id,
  label,
  interactive
}) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      // A non-interactive graphic is a single labelled image to assistive tech;
      // an interactive one is a labelled group whose parts are focusable buttons
      // (role="img" would hide those children from the accessibility tree).
      const role = interactive ? 'group' : 'img';
      ref.current.innerHTML = svgBase
        .replace(
          '<svg',
          `<svg id="${id}" width="100%" height="100%" role="${role}" aria-label="${label}"`
        )
        // The base photo is decorative relative to the labelled root.
        .replace('<image', '<image aria-hidden="true"');
    }
  }, [id, label, interactive]);

  return <div ref={ref} style={{ width: '100%', height: '100%' }} />;
};

export default CarGraphic;
