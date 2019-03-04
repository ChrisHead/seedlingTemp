import jpegJs from "jpeg-js"
import fs from "fs"
import { promisify } from "util"
import path from "path"

const readdir = promisify(fs.readdir)
const readFile = promisify(fs.readFile)

const testFolder = "src/testData/"

async function readJpeg(filePath: string) {
  const img = await readFile(filePath)
  const rawData = jpegJs.decode(img, true)

  const size = Array.from(rawData.data).length

  const removedAlpha: number[] = []
  for (let i = 0; i < size; i++) {
    if (i % 4 === 3) {
      continue
    }
    removedAlpha.push(rawData.data[i])
  }
  return removedAlpha
}

function normalizeJpeg(data: number[]) {
  return data.map(datum => datum / 255)
}
async function convert() {
  const folderName = "healthy"
  const paths = await readdir(path.join(testFolder, folderName))

  const isFile = file => file !== "." && file !== ".."

  const validPaths = paths.filter(isFile)

  for (const f of validPaths) {
    const fullPath = path.join(testFolder, folderName, f)
    const jpeg = await readJpeg(fullPath)
    // console.log(jpeg)
    const imageData = normalizeJpeg(jpeg)

    const payload = Buffer.from(jpeg)
    fs.writeFileSync(
      path.join(testFolder, `${folderName}Parsed`, f + ".json"),
      payload
    )
  }
}

convert()
