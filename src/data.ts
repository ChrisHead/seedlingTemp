import "@tensorflow/tfjs-node"

import * as tf from "@tensorflow/tfjs"
import { shuffle } from "@tensorflow/tfjs-core/dist/util"
import fs from "fs"
import jpegJs from "jpeg-js"
import path from "path"
import util from "util"

export const IMAGE_H = 256
export const IMAGE_W = 256

const testFolder = "src/testData/"
const readdir = util.promisify(fs.readdir)
const readFile = util.promisify(fs.readFile)

interface IStructure {
  data: number[]
  label: number
}

interface ILabeledFile {
  name: string
  label: number
}

export interface IBatch {
  tensors: ITensor<4>
  labels: ITensor<2>
  batchSize: number
}

type ITensor<n extends number = 0> = n extends 4
  ? tf.Tensor<tf.Rank.R4>
  : n extends 2
  ? tf.Tensor<tf.Rank.R2>
  : tf.Tensor

export class ImageData {
  batchSize: number
  validationBatchSize: number
  validationDataPerc: number
  testDataPerc: number

  curIndex = 0
  validationStart = 0

  private loadedFilenames?: {
    training: ILabeledFile[]
    test: ILabeledFile[]
    validation: ILabeledFile[]
  }

  constructor({
    batchSize = 32,
    validationBatchSize = 32,
    validationDataPerc = 0.1,
    testDataPerc = 0.2,
  }: {
    batchSize?: number
    validationBatchSize?: number
    validationDataPerc?: number
    testDataPerc?: number
  }) {
    this.batchSize = batchSize
    this.validationBatchSize = validationBatchSize
    this.validationDataPerc = validationDataPerc
    this.testDataPerc = testDataPerc
  }

  async nextBatch(): Promise<IBatch | undefined> {
    const filenames = await this.filenames()
    const batchNames = filenames.training.slice(
      this.curIndex,
      this.curIndex + this.batchSize
    )
    this.curIndex += this.batchSize

    if (batchNames.length === 0) {
      return undefined
    }

    return this.loadFilesAndTensors(batchNames)
  }

  async getValidationBatch() {
    const filenames = await this.filenames()
    const validationNames = filenames.validation
    const selectedNames = validationNames.slice(
      this.validationStart,
      this.validationStart + this.validationBatchSize
    )
    this.validationStart += this.validationBatchSize
    this.validationStart = this.validationStart % filenames.validation.length
    return this.loadFilesAndTensors(selectedNames)
  }

  async getTestBatch() {
    const filenames = await this.filenames()
    const validationNames = filenames.test
    return this.loadFilesAndTensors(validationNames)
  }

  async loadFilesAndTensors(names: ILabeledFile[]) {
    const structs: IStructure[] = []
    console.time("reading jpeg")

    for (const labeledFile of names) {
      const jpeg = Array.from<number>(await readFile(labeledFile.name))
      // const jpeg = require("../" + labeledFile.name) as number[]
      // const jpeg = await this.readJpeg(labeledFile.name)
      const imageData = this.normalizeJpeg(jpeg)
      structs.push({ data: imageData, label: labeledFile.label })
    }

    console.timeEnd("reading jpeg")

    console.time("creating tensors")
    const data = ([] as number[]).concat(...structs.map(struct => struct.data))
    const labels = structs.map(struct => struct.label)

    const tensors = tf.tensor4d(data, [structs.length, IMAGE_H, IMAGE_W, 3])
    const labelTensors = tf.tensor2d(labels, [structs.length, 3])
    console.timeEnd("creating tensors")

    return { tensors, labels: labelTensors, batchSize: structs.length }
  }

  async filenames() {
    if (!this.loadedFilenames) {
      const filenames = await this.loadFilenames()
      shuffle(filenames)
      this.loadedFilenames = this.segmentFilenames(filenames)
    }
    return this.loadedFilenames!
  }

  async loadFilenames() {
    const diseaseOne = "diseaseOneParsed"
    const diseaseTwo = "diseaseTwoParsed"
    const healthy = "healthyParsed"
    const diseaseOneFileNames = await readdir(path.join(testFolder, diseaseOne))
    const diseaseTwoFileNames = await readdir(path.join(testFolder, diseaseTwo))
    const healthyFileNames = await readdir(path.join(testFolder, healthy))

    const isFile = file => file !== "." && file !== ".."

    const diseaseOneFilePaths = diseaseOneFileNames
      .filter(isFile)
      .map(name => path.join(testFolder, diseaseOne, name))
    const diseaseTwoFilePaths = diseaseTwoFileNames
      .filter(isFile)
      .map(name => path.join(testFolder, diseaseTwo, name))
    const healthyFilePaths = healthyFileNames
      .filter(isFile)
      .map(name => path.join(testFolder, healthy, name))

    const dis1 = diseaseOneFilePaths.map(name => ({
      name,
      label: [0.9, 0.1, 0.1],
    }))
    const dis2 = diseaseTwoFilePaths.map(name => ({
      name,
      label: [0.1, 0.9, 0.1],
    }))
    const heal = healthyFilePaths.map(name => ({
      name,
      label: [0.1, 0.1, 0.9],
    }))

    return dis1.concat(heal).concat(dis2)
  }

  segmentFilenames(names) {
    const len = names.length
    const testCnt = Math.floor(len * this.testDataPerc)
    const validCnt = Math.floor(len * this.validationDataPerc)

    const test = names.slice(0, testCnt)
    const validation = names.slice(testCnt, testCnt + validCnt)
    const training = names.slice(testCnt + validCnt)

    return { test, validation, training }
  }

  async loadTestImage() {
    const testImage = await readdir(path.join(testFolder, "Test"))
    const isFile = file => file !== "." && file !== ".."
    const corrFilePath = testImage
      .filter(isFile)
      .map(name => path.join(testFolder, "Test", name))
    const jpeg = await this.readJpeg(corrFilePath[0])
    const imageData = await this.normalizeJpeg(jpeg)
    const tensor = tf.tensor4d(imageData, [1, IMAGE_H, IMAGE_W, 3])
    return tensor
  }

  async readJpeg(filePath: string) {
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

  normalizeJpeg(data: number[]) {
    return data.map(datum => datum / 255)
  }
}
