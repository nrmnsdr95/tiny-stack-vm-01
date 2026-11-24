import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join as pathJoin } from 'node:path';

// Opcode table: [code, immSize, pops, pushes]
const INSTRUCTION_TABLE: [number, number, number, number][] = [
  [0x00, 0, 0, 0], [0x01, 1, 0, 1], [0x02, 4, 0, 1], [0x03, 0, 1, 0],
  [0x04, 0, 1, 2], [0x05, 0, 2, 2], [0x06, 0, 2, 3], [0x07, 0, 3, 3],
  [0x10, 0, 2, 1], [0x11, 0, 2, 1], [0x12, 0, 2, 1], [0x13, 0, 2, 1],
  [0x14, 0, 2, 1], [0x15, 0, 2, 1], [0x16, 0, 2, 1], [0x17, 0, 1, 1],
  [0x18, 0, 2, 1], [0x19, 0, 2, 1], [0x1a, 0, 2, 1], [0x1b, 0, 1, 1],
  [0x1c, 0, 2, 1], [0x1d, 0, 2, 1], [0x1e, 0, 2, 1], [0x20, 0, 2, 1],
  [0x21, 0, 2, 1], [0x22, 0, 2, 1], [0x30, 2, 0, 0], [0x31, 2, 1, 0],
  [0x32, 2, 1, 0], [0x33, 2, 0, 0], [0x34, 0, 0, 0], [0x40, 0, 1, 1],
  [0x41, 0, 2, 0], [0x50, 0, 1, 0], [0x51, 0, 1, 0], [0x52, 0, 0, 1],
  [0x53, 0, 1, 0],
];

const IMM_SIZES = new Int8Array(256).fill(-1);
const POP_COUNTS = new Uint8Array(256);
const PUSH_COUNTS = new Uint8Array(256);
for (const [code, imm, pops, pushes] of INSTRUCTION_TABLE) {
  IMM_SIZES[code] = imm;
  POP_COUNTS[code] = pops;
  PUSH_COUNTS[code] = pushes;
}

const textEncoder = new TextEncoder();

export function executeVM(program: Uint8Array, input: Uint8Array) {
  const codeLen = program.length;
  const dataStack = new Int32Array(256);
  const callStack = new Int32Array(64);
  const memory = new Int32Array(256);
  const outBuf = new Uint8Array(65536);

  let stackDepth = 0, callDepth = 0, outPtr = 0, inPtr = 0, pcReg = 0, steps = 0;
  const terminate = (exitCode: number) => ({ stdout: outBuf.subarray(0, outPtr), exit: exitCode });

  while (true) {
    if (steps === 1000000) return terminate(10);
    if (pcReg < 0 || pcReg >= codeLen) return terminate(9);

    const opcode = program[pcReg];
    const immLen = IMM_SIZES[opcode];
    if (immLen === -1) return terminate(5);
    if (pcReg + immLen >= codeLen) return terminate(9);

    steps++;
    const nextPc = pcReg + 1 + immLen;
    if (stackDepth < POP_COUNTS[opcode]) return terminate(2);
    if (stackDepth - POP_COUNTS[opcode] + PUSH_COUNTS[opcode] > 256) return terminate(3);
    pcReg = nextPc;

    // Fast path: binary arithmetic and bitwise / comparison operations
    if (opcode >= 0x10 && opcode <= 0x22 && opcode !== 0x17 && opcode !== 0x1b) {
      const right = dataStack[stackDepth - 1];
      const left = dataStack[stackDepth - 2];
      let res = 0;
      switch (opcode) {
        case 0x10: res = (left + right) | 0; break;
        case 0x11: res = (left - right) | 0; break;
        case 0x12: res = Math.imul(left, right); break;
        case 0x13: if (right === 0) return terminate(4); res = left === -2147483648 && right === -1 ? left : (left / right) | 0; break;
        case 0x14: if (right === 0) return terminate(4); res = right === -1 ? 0 : (left % right) | 0; break;
        case 0x15: if (right === 0) return terminate(4); res = Math.floor((left >>> 0) / (right >>> 0)) | 0; break;
        case 0x16: if (right === 0) return terminate(4); res = ((left >>> 0) % (right >>> 0)) | 0; break;
        case 0x18: res = left & right; break;
        case 0x19: res = left | right; break;
        case 0x1a: res = left ^ right; break;
        case 0x1c: res = left << (right & 31); break;
        case 0x1d: res = left >> (right & 31); break;
        case 0x1e: res = (left >>> (right & 31)) | 0; break;
        case 0x20: res = left === right ? 1 : 0; break;
        case 0x21: res = left < right ? 1 : 0; break;
        case 0x22: res = (left >>> 0) < (right >>> 0) ? 1 : 0; break;
      }
      stackDepth--;
      dataStack[stackDepth - 1] = res;
      continue;
    }

    switch (opcode) {
      case 0x00: return terminate(0);
      case 0x01: dataStack[stackDepth++] = (program[nextPc - 1] << 24) >> 24; break;
      case 0x02:
        dataStack[stackDepth++] = program[nextPc - 4] | (program[nextPc - 3] << 8) | (program[nextPc - 2] << 16) | (program[nextPc - 1] << 24);
        break;
      case 0x03: stackDepth--; break;
      case 0x04: dataStack[stackDepth] = dataStack[stackDepth - 1]; stackDepth++; break;
      case 0x05: { const t = dataStack[stackDepth - 1]; dataStack[stackDepth - 1] = dataStack[stackDepth - 2]; dataStack[stackDepth - 2] = t; break; }
      case 0x06: dataStack[stackDepth] = dataStack[stackDepth - 2]; stackDepth++; break;
      case 0x07: { const b = dataStack[stackDepth - 3]; dataStack[stackDepth - 3] = dataStack[stackDepth - 2]; dataStack[stackDepth - 2] = dataStack[stackDepth - 1]; dataStack[stackDepth - 1] = b; break; }
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
        const addr = dataStack[stackDepth - 1], val = dataStack[stackDepth - 2];
        if (addr < 0 || addr >= 256) return terminate(6);
        stackDepth -= 2;
        memory[addr] = val;
        break;
      }
      case 0x50: {
        const enc = textEncoder.encode(String(dataStack[stackDepth - 1]));
        if (outPtr + enc.length > 65536) return terminate(11);
        stackDepth--;
        outBuf.set(enc, outPtr);
        outPtr += enc.length;
        break;
      }
      case 0x51:
        if (outPtr + 1 > 65536) return terminate(11);
        outBuf[outPtr++] = dataStack[--stackDepth] & 255;
        break;
      case 0x52: dataStack[stackDepth++] = inPtr < input.length ? input[inPtr++] : -1; break;
      case 0x53: return terminate(dataStack[--stackDepth] & 255);
    }
  }
}

const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

for (const id of readdirSync(inputDir).sort()) {
  const prog = new Uint8Array(readFileSync(pathJoin(inputDir, id, 'program.bin')));
  const inp = new Uint8Array(readFileSync(pathJoin(inputDir, id, 'input.bin')));
  const res = executeVM(prog, inp);

  mkdirSync(pathJoin(outputDir, id), { recursive: true });
  writeFileSync(pathJoin(outputDir, id, 'stdout.bin'), res.stdout);
  writeFileSync(pathJoin(outputDir, id, 'exit.txt'), `${res.exit}\n`);
}
