import React from 'react';
import interactSource from '../services/svg-interact.service.ts?raw';
import CodeSnippet from './CodeSnippet';

const InteractCodeSnippet: React.FC = () => (
  <CodeSnippet code={interactSource} />
);

export default InteractCodeSnippet;
