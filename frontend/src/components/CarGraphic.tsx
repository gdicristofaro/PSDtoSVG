import React, { useRef, useEffect } from 'react';
import carGraphicRaw from './CarGraphic.svg?raw';

const svgBase = carGraphicRaw.replace(/^<\?xml[^?]*\?>\s*/s, '');

const CarGraphic: React.FC<{id: string}> = ({ id }) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.innerHTML = svgBase.replace('<svg', `<svg id="${id}" width="100%" height="100%"`);
    }
  }, [id]);

  return <div ref={ref} style={{ width: '100%', height: '100%' }} />;
};

export default CarGraphic;
