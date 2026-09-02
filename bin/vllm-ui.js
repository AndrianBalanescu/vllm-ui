#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');

const scriptPath = path.join(__dirname, '..', 'vllm_ui', 'server.py');
const args = [scriptPath, ...process.argv.slice(2)];

const child = spawn('python3', args, { stdio: 'inherit' });

child.on('error', (err) => {
  if (err.code === 'ENOENT') {
    console.error('[vllm-ui] Error: Python 3 is required to run vllm-ui.');
  } else {
    console.error('[vllm-ui] Error:', err.message);
  }
  process.exit(1);
});

child.on('exit', (code) => {
  process.exit(code || 0);
});
