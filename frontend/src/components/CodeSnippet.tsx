import React from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { darcula } from 'react-syntax-highlighter/dist/esm/styles/hljs';

const CodeSnippet: React.FC<{code: string}> = (props: { code: string }) => (
      <SyntaxHighlighter 
        language="javascript"
        style={darcula} 
        customStyle={{
            padding: 0,
            margin: 0,
            background: "transparent",
            width: "max-content"
            }}>
        {props.code}
      </SyntaxHighlighter>
);

export default CodeSnippet;
