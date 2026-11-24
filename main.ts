import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join as pathJoin } from 'node:path';

// High-performance compact table-driven VM execution engine
// Table mapping opcode to [immediateByteCount, popCount, pushCount]
const OP_DEFS: Record<number, [number, number, number]> = {
  0x00: [0, 0, 0], 0x01: [1, 0, 1], 0x02: [4, 0, 1], 0x03: [0, 1, 0],
  0x04: [0, 1, 2], 0x05: [0, 2, 2], 0x06: [0, 2, 3], 0x07: [0, 3, 3],
  0x10: [0, 2, 1], 0x11: [0, 2, 1], 0x12: [0, 2, 1], 0x13: [0, 2, 1],
  0x14: [0, 2, 1], 0x15: [0, 2, 1], 0x16: [0, 2, 1], 0x17: [0, 1, 1],
  0x18: [0, 2, 1], 0x19: [0, 2, 1], 0x1a: [0, 2, 1], 0x1b: [0, 1, 1],
  0x1c: [0, 2, 1], 0x1d: [0, 2, 1], 0x1e: [0, 2, 1], 0x20: [0, 2, 1],
  0x21: [0, 2, 1], 0x22: [0, 2, 1], 0x30: [2, 0, 0], 0x31: [2, 1, 0],
  0x32: [2, 1, 0], 0x33: [2, 0, 0], 0x34: [0, 0, 0], 0x40: [0, 1, 1],
  0x41: [0, 2, 0], 0x50: [0, 1, 0], 0x51: [0, 1, 0], 0x52: [0, 0, 1],
  0x53: [0, 1, 0]
};

const BINARY_FUNCS = new Map<number, (x: number, y: number) => number>([
  [0x10, (x, y) => (x + y) | 0],
  [0x11, (x, y) => (x - y) | 0],
  [0x12, (x, y) => Math.imul(x, y)],
  [0x18, (x, y) => x & y],
  [0x19, (x, y) => x | y],
  [0x1a, (x, y) => x ^ y],
  [0x1c, (x, y) => x << (y & 31)],
  [0x1d, (x, y) => x >> (y & 31)],
  [0x1e, (x, y) => (x >>> (y & 31)) | 0],
  [0x20, (x, y) => (x === y ? 1 : 0)],
  [0x21, (x, y) => (x < y ? 1 : 0)],
  [0x22, (x, y) => ((x >>> 0) < (y >>> 0) ? 1 : 0)],
]);

export function executeVM(program: Uint8Array, input: Uint8Array) {
  const codeLen = program.length;
  const dataStack = new Int32Array(256);
  const callStack = new Int32Array(64);
  const memory = new Int32Array(256);
  const outBuf = new Uint8Array(65536);

  let stackDepth = 0, callDepth = 0, outPtr = 0, inPtr = 0, pcReg = 0, steps = 0;
  const terminate = (exitCode: number) => ({ stdout: outBuf.subarray(0, outPtr), exit: exitCode });
  const pushVal = (v: number) => { dataStack[stackDepth++] = v; };

  while (true) {
    if (steps === 1000000) return terminate(10);
    if (pcReg < 0 || pcReg >= codeLen) return terminate(9);

    const opcode = program[pcReg];
    const def = OP_DEFS[opcode];
    if (!def) return terminate(5);
    const [immLen, pops, pushes] = def;
    if (pcReg + immLen >= codeLen) return terminate(9);

    steps++;
    const nextPc = pcReg + 1 + immLen;
    if (stackDepth < pops) return terminate(2);
    if (stackDepth - pops + pushes > 256) return terminate(3);
    pcReg = nextPc;

    const binFn = BINARY_FUNCS.get(opcode);
    if (binFn) {
      const right = dataStack[stackDepth - 1];
      const left = dataStack[stackDepth - 2];
      stackDepth--;
      dataStack[stackDepth - 1] = binFn(left, right);
      continue;
    }

    if (opcode >= 0x13 && opcode <= 0x16) {
      const right = dataStack[stackDepth - 1];
      const left = dataStack[stackDepth - 2];
      if (right === 0) return terminate(4);
      let res = 0;
      if (opcode === 0x13) res = (left === -0x80000000 && right === -1) ? left : ((left / right) | 0);
      else if (opcode === 0x14) res = (right === -1) ? 0 : ((left % right) | 0);
      else if (opcode === 0x15) res = Math.floor((left >>> 0) / (right >>> 0)) | 0;
      else if (opcode === 0x16) res = ((left >>> 0) % (right >>> 0)) | 0;
      stackDepth--;
      dataStack[stackDepth - 1] = res;
      continue;
    }

    switch (opcode) {
      case 0x00: return terminate(0);
      case 0x01: pushVal((program[nextPc - 1] << 24) >> 24); break;
      case 0x02: {
        const p = nextPc - 4;
        pushVal(program[p] | (program[p + 1] << 8) | (program[p + 2] << 16) | (program[p + 3] << 24));
        break;
      }
      case 0x03: stackDepth--; break;
      case 0x04: pushVal(dataStack[stackDepth - 1]); break;
      case 0x05: {
        const top = dataStack[stackDepth - 1];
        dataStack[stackDepth - 1] = dataStack[stackDepth - 2];
        dataStack[stackDepth - 2] = top;
        break;
      }
      case 0x06: pushVal(dataStack[stackDepth - 2]); break;
      case 0x07: {
        const third = dataStack[stackDepth - 3];
        dataStack[stackDepth - 3] = dataStack[stackDepth - 2];
        dataStack[stackDepth - 2] = dataStack[stackDepth - 1];
        dataStack[stackDepth - 1] = third;
        break;
      }
      case 0x17: dataStack[stackDepth - 1] = (-dataStack[stackDepth - 1]) | 0; break;
      case 0x1b: dataStack[stackDepth - 1] = ~dataStack[stackDepth - 1]; break;
      case 0x30:
      case 0x31:
      case 0x32: {
        const offset = ((program[nextPc - 2] | (program[nextPc - 1] << 8)) << 16) >> 16;
        if (opcode === 0x30 || (opcode === 0x31 ? dataStack[--stackDepth] === 0 : dataStack[--stackDepth] !== 0)) pcReg = nextPc + offset;
        break;
      }
      case 0x33:
        if (callDepth === 64) return terminate(7);
        callStack[callDepth++] = nextPc;
        pcReg = program[nextPc - 2] | (program[nextPc - 1] << 8);
        break;
      case 0x34:
        if (callDepth === 0) return terminate(8);
        pcReg = callStack[--callDepth];
        break;
      case 0x40: {
        const addr = dataStack[stackDepth - 1];
        if (addr < 0 || addr >= 256) return terminate(6);
        dataStack[stackDepth - 1] = memory[addr];
        break;
      }
      case 0x41: {
        const addr = dataStack[--stackDepth], val = dataStack[--stackDepth];
        if (addr < 0 || addr >= 256) return terminate(6);
        memory[addr] = val;
        break;
      }
      case 0x50: {
        const numStr = String(dataStack[stackDepth - 1]);
        if (outPtr + numStr.length > 65536) return terminate(11);
        stackDepth--;
        for (let idx = 0; idx < numStr.length; idx++) outBuf[outPtr++] = numStr.charCodeAt(idx);
        break;
      }
      case 0x51:
        if (outPtr + 1 > 65536) return terminate(11);
        outBuf[outPtr++] = dataStack[--stackDepth] & 255;
        break;
      case 0x52: pushVal(inPtr < input.length ? input[inPtr++] : -1); break;
      case 0x53: return terminate(dataStack[--stackDepth] & 255);
    }
  }
}

const inputFolder = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputFolder = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

for (const entryId of readdirSync(inputFolder).sort()) {
  const progBytes = new Uint8Array(readFileSync(pathJoin(inputFolder, entryId, 'program.bin')));
  const inpBytes = new Uint8Array(readFileSync(pathJoin(inputFolder, entryId, 'input.bin')));
  const vmResult = executeVM(progBytes, inpBytes);
  mkdirSync(pathJoin(outputFolder, entryId), { recursive: true });
  writeFileSync(pathJoin(outputFolder, entryId, 'stdout.bin'), vmResult.stdout);
  writeFileSync(pathJoin(outputFolder, entryId, 'exit.txt'), `${vmResult.exit}\n`);
}
