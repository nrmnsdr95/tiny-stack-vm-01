import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join as pathJoin } from 'node:path';

const OP_SPECS: ([number, number, number] | null)[] = new Array(256).fill(null);
[
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
].forEach(([c, i, po, pu]) => { OP_SPECS[c] = [i, po, pu]; });

export class TinyStackVM {
  code: Uint8Array;
  stream: Uint8Array;
  ds = new Int32Array(256);
  cs = new Int32Array(64);
  ram = new Int32Array(256);
  out = new Uint8Array(65536);
  sp = 0;
  csp = 0;
  outCount = 0;
  inPos = 0;
  pc = 0;
  stepCount = 0;

  constructor(code: Uint8Array, stream: Uint8Array) {
    this.code = code;
    this.stream = stream;
  }

  private result(exitCode: number) {
    return { stdout: this.out.subarray(0, this.outCount), exit: exitCode };
  }

  private popValue(): number {
    return this.ds[--this.sp];
  }

  private performMath(op: number, a: number, b: number): number {
    switch (op) {
      case 0x10: return (a + b) | 0;
      case 0x11: return (a - b) | 0;
      case 0x12: return Math.imul(a, b);
      case 0x18: return a & b;
      case 0x19: return a | b;
      case 0x1a: return a ^ b;
      case 0x1c: return a << (b & 31);
      case 0x1d: return a >> (b & 31);
      case 0x1e: return (a >>> (b & 31)) | 0;
      case 0x20: return a === b ? 1 : 0;
      case 0x21: return a < b ? 1 : 0;
      case 0x22: return (a >>> 0) < (b >>> 0) ? 1 : 0;
      default: return 0;
    }
  }

  run(): { stdout: Uint8Array; exit: number } {
    const code = this.code;
    const len = code.length;
    const ds = this.ds;

    while (true) {
      if (this.stepCount === 1000000) return this.result(10);
      if (this.pc < 0 || this.pc >= len) return this.result(9);

      const op = code[this.pc];
      const spec = OP_SPECS[op];
      if (!spec) return this.result(5);

      const [immSize, pops, pushes] = spec;
      if (this.pc + immSize >= len) return this.result(9);

      this.stepCount++;
      const nextAddress = this.pc + 1 + immSize;

      if (this.sp < pops) return this.result(2);
      if (this.sp - pops + pushes > 256) return this.result(3);
      this.pc = nextAddress;

      // Binary math / bitwise
      if ((op >= 0x10 && op <= 0x12) || (op >= 0x18 && op <= 0x1a) || (op >= 0x1c && op <= 0x1e) || (op >= 0x20 && op <= 0x22)) {
        const y = ds[this.sp - 1];
        const x = ds[this.sp - 2];
        this.sp -= 2;
        ds[this.sp++] = this.performMath(op, x, y);
        continue;
      }

      // Divisions and Modulos
      if (op >= 0x13 && op <= 0x16) {
        const y = ds[this.sp - 1], x = ds[this.sp - 2];
        if (y === 0) return this.result(4);
        let calc = 0;
        if (op === 0x13) calc = (x === -2147483648 && y === -1) ? x : ((x / y) | 0);
        else if (op === 0x14) calc = (y === -1) ? 0 : ((x % y) | 0);
        else if (op === 0x15) calc = Math.floor((x >>> 0) / (y >>> 0)) | 0;
        else if (op === 0x16) calc = ((x >>> 0) % (y >>> 0)) | 0;
        this.sp -= 2;
        ds[this.sp++] = calc;
        continue;
      }

      switch (op) {
        case 0x00: return this.result(0);
        case 0x01: ds[this.sp++] = (code[nextAddress - 1] << 24) >> 24; break;
        case 0x02: {
          const off = nextAddress - 4;
          ds[this.sp++] = code[off] | (code[off + 1] << 8) | (code[off + 2] << 16) | (code[off + 3] << 24);
          break;
        }
        case 0x03: this.sp--; break;
        case 0x04: ds[this.sp] = ds[this.sp - 1]; this.sp++; break;
        case 0x05: { const t = ds[this.sp - 1]; ds[this.sp - 1] = ds[this.sp - 2]; ds[this.sp - 2] = t; break; }
        case 0x06: ds[this.sp] = ds[this.sp - 2]; this.sp++; break;
        case 0x07: { const t = ds[this.sp - 3]; ds[this.sp - 3] = ds[this.sp - 2]; ds[this.sp - 2] = ds[this.sp - 1]; ds[this.sp - 1] = t; break; }
        case 0x17: ds[this.sp - 1] = (-ds[this.sp - 1]) | 0; break;
        case 0x1b: ds[this.sp - 1] = ~ds[this.sp - 1]; break;
        case 0x30:
        case 0x31:
        case 0x32: {
          const delta = ((code[nextAddress - 2] | (code[nextAddress - 1] << 8)) << 16) >> 16;
          if (op === 0x30 || (op === 0x31 ? ds[--this.sp] === 0 : ds[--this.sp] !== 0)) this.pc = nextAddress + delta;
          break;
        }
        case 0x33:
          if (this.csp === 64) return this.result(7);
          this.cs[this.csp++] = nextAddress;
          this.pc = code[nextAddress - 2] | (code[nextAddress - 1] << 8);
          break;
        case 0x34:
          if (this.csp === 0) return this.result(8);
          this.pc = this.cs[--this.csp];
          break;
        case 0x40: {
          const target = ds[this.sp - 1];
          if (target < 0 || target >= 256) return this.result(6);
          ds[this.sp - 1] = this.ram[target];
          break;
        }
        case 0x41: {
          const loc = ds[--this.sp], datum = ds[--this.sp];
          if (loc < 0 || loc >= 256) return this.result(6);
          this.ram[loc] = datum;
          break;
        }
        case 0x50: {
          const digits = String(ds[--this.sp]);
          if (this.outCount + digits.length > 65536) return this.result(11);
          for (let k = 0; k < digits.length; k++) this.out[this.outCount++] = digits.charCodeAt(k);
          break;
        }
        case 0x51:
          if (this.outCount + 1 > 65536) return this.result(11);
          this.out[this.outCount++] = this.popValue() & 255;
          break;
        case 0x52:
          ds[this.sp++] = this.inPos < this.stream.length ? this.stream[this.inPos++] : -1;
          break;
        case 0x53:
          return this.result(this.popValue() & 255);
      }
    }
  }
}

// Drive execution over input batches
const sourcePath = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const targetPath = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

for (const name of readdirSync(sourcePath).sort()) {
  const codeBytes = new Uint8Array(readFileSync(pathJoin(sourcePath, name, 'program.bin')));
  const streamBytes = new Uint8Array(readFileSync(pathJoin(sourcePath, name, 'input.bin')));
  const vmInstance = new TinyStackVM(codeBytes, streamBytes);
  const outcome = vmInstance.run();

  mkdirSync(pathJoin(targetPath, name), { recursive: true });
  writeFileSync(pathJoin(targetPath, name, 'stdout.bin'), outcome.stdout);
  writeFileSync(pathJoin(targetPath, name, 'exit.txt'), `${outcome.exit}\n`);
}
