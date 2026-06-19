import React from 'react';
import animateSource from '../services/svg-animation.service.ts?raw';
import CodeSnippet from './CodeSnippet';

const AnimateCodeSnippet: React.FC = () => (
  <CodeSnippet code={animateSource} />
);

export default AnimateCodeSnippet;
