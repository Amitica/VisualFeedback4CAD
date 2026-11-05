import { deflateSync } from "node:zlib";

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function crc32(bytes: Uint8Array): number {
	let crc = 0xffffffff;
	for (const byte of bytes) {
		crc ^= byte;
		for (let bit = 0; bit < 8; bit += 1) {
			crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
		}
	}
	return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Buffer): Buffer {
	const typeBytes = Buffer.from(type, "ascii");
	const result = Buffer.alloc(12 + data.length);
	result.writeUInt32BE(data.length, 0);
	typeBytes.copy(result, 4);
	data.copy(result, 8);
	result.writeUInt32BE(crc32(Buffer.concat([typeBytes, data])), 8 + data.length);
	return result;
}

/** Build a complete, decodable RGBA PNG using only Node built-ins. */
export function makePng(width: number, height: number, marker = 1): Buffer {
	if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
		throw new Error("PNG fixture dimensions must be positive integers");
	}
	const ihdr = Buffer.alloc(13);
	ihdr.writeUInt32BE(width, 0);
	ihdr.writeUInt32BE(height, 4);
	ihdr[8] = 8;
	ihdr[9] = 6;
	const scanlines = Buffer.alloc(height * (1 + width * 4));
	for (let row = 0; row < height; row += 1) {
		const offset = row * (1 + width * 4);
		scanlines[offset] = 0;
		for (let column = 0; column < width; column += 1) {
			const pixel = offset + 1 + column * 4;
			scanlines[pixel] = marker;
			scanlines[pixel + 1] = (marker + column) & 0xff;
			scanlines[pixel + 2] = (marker + row) & 0xff;
			scanlines[pixel + 3] = 0xff;
		}
	}
	return Buffer.concat([
		PNG_SIGNATURE,
		chunk("IHDR", ihdr),
		chunk("IDAT", deflateSync(scanlines)),
		chunk("IEND", Buffer.alloc(0)),
	]);
}

/** Rewrite IHDR dimensions and its CRC without allocating the claimed raster. */
export function rewritePngDimensions(png: Buffer, width: number, height: number): Buffer {
	const result = Buffer.from(png);
	result.writeUInt32BE(width, 16);
	result.writeUInt32BE(height, 20);
	result.writeUInt32BE(crc32(result.subarray(12, 29)), 29);
	return result;
}

export function corruptIhdrCrc(png: Buffer): Buffer {
	const result = Buffer.from(png);
	result[32] = (result[32] ?? 0) ^ 0xff;
	return result;
}

export function fakeIhdrOnly(width: number, height: number): Buffer {
	const result = Buffer.alloc(25);
	PNG_SIGNATURE.copy(result, 0);
	result.write("IHDR", 12, "ascii");
	result.writeUInt32BE(width, 16);
	result.writeUInt32BE(height, 20);
	return result;
}
