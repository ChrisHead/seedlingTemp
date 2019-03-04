export interface IClassificationResult {
  final: {
    type: string
    confidence: number
  }
  raw: {
    [key: string]: number
  }
}
