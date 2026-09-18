package common

import deepwit.cnn.AffineConv2DLayer
import deepwit.embedder.PositionalEncoding.sinusoidal2D
import dimwit.*
import munit.FunSuite

class ImageToPatchEmbedderSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  trait Width derives Label
  trait Height derives Label
  trait Channel derives Label
  trait PatchEmbedding derives Label

  private val patch = Shape(Axis[Width] -> 16, Axis[Height] -> 16, Axis[Channel] -> 1)
  private val embeddingExtent = Axis[PatchEmbedding] -> 4

  private def embedder(key: Key) =
    ImageToPatchEmbedder(Axis[Width], Axis[Height], Axis[Channel], ImageToPatchEmbedder.Params.xavierUniform(embeddingExtent, key))

  private def drawing(width: Int, height: Int) =
    val shape = Shape(Axis[Width] -> width, Axis[Height] -> height, Axis[Channel] -> 1)
    Tensor(shape, VType[Float32]).fromArray(Array.tabulate(shape.dimensions.product)(i => math.sin(i.toFloat)))

  test("produces one embedded patch per patch of the drawing"):
    val patches = embedder(Random.Key(42))(drawing(64, 32))
    assertEquals(patches.shape(Axis[Width |*| Height]), 4 * 2)
    assertEquals(patches.shape(Axis[PatchEmbedding]), 4)

  test("distinguishes patches through the positional encoding"):
    val flat = Tensor(Shape(Axis[Width] -> 32, Axis[Height] -> 32, Axis[Channel] -> 1), VType[Float32]).fill(0.5f)
    val patches = embedder(Random.Key(42))(flat)
    val first = patches.slice(Axis[Width |*| Height].at(0))
    val last = patches.slice(Axis[Width |*| Height].at(3))
    assert((first - last).abs.max.item > 1e-3f, "a constant drawing gives the same patch everywhere but for the encoding")

  test("equals a convolution whose stride is its kernel, plus the positional encoding"):
    // A non-square drawing, so that a mix-up of the in-patch order or the patch grid would show.
    val params = ImageToPatchEmbedder.Params.xavierUniform(embeddingExtent, Random.Key(7))
    val image = drawing(48, 32)

    val kernel = params.embed.weight.unflatten(Axis[PatchFeature], patch)
    val conv = AffineConv2DLayer(AffineConv2DLayer.Params(kernel, params.embed.bias), stride = (patch.extent(Axis[Width]), patch.extent(Axis[Height])))
    val convolved = conv(image)
    val expected = (convolved + sinusoidal2D(convolved.shape)).flatten((Axis[Width], Axis[Height]))

    val actual = ImageToPatchEmbedder(Axis[Width], Axis[Height], Axis[Channel], params)(image)
    assert((actual - expected).abs.max.item < 1e-5f)

  test("xavierUniform maps the flattened pixels of a patch to the embedding"):
    val params = ImageToPatchEmbedder.Params.xavierUniform(embeddingExtent, Random.Key(42))
    assertEquals(params.embed.weight.shape(Axis[PatchFeature]), 16 * 16)
    assertEquals(params.embed.weight.shape(Axis[PatchEmbedding]), 4)
    assert(params.embed.bias.abs.max.item == 0f)
