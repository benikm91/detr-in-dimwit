package documentEncoder

import dimwit.*
import deepwit.base.AffineLayer
import deepwit.embedder.PositionalEncoding.sinusoidal2D
import dimwit.Label as Λ

trait PatchFeature derives Label

/** Cuts a drawing into non-overlapping 16x16 patches, embeds each with one affine map, adds a 2D
  * sinusoidal positional encoding and flattens the grid into a sequence. The drawing must cut into
  * whole patches.
  */
class ImageToPatchEmbedder[PatchEmbedding: Λ, V: IsFloating](
    params: ImageToPatchEmbedder.Params[PatchEmbedding, V]
) extends (Tensor3[Width, Height, Channel, V] => Tensor2[Patch, PatchEmbedding, V]):

  import ImageToPatchEmbedder.patchSize

  private val embed = AffineLayer(params.embed)

  override def apply(img: Tensor3[Width, Height, Channel, V]): Tensor2[Patch, PatchEmbedding, V] =
    trait PatchX derives Label
    trait PatchY derives Label
    val patches = img
      .unflatten(Axis[Width], Shape(Axis[PatchX] -> img.shape(Axis[Width]) / patchSize, Axis[Width] -> patchSize))
      .unflatten(Axis[Height], Shape(Axis[PatchY] -> img.shape(Axis[Height]) / patchSize, Axis[Height] -> patchSize))
    val grid = patches
      .vmap(Axis[PatchX])(_.vmap(Axis[PatchY])(p => embed(p.flatten.relabelTo(Axis[PatchFeature]))))
    (grid + sinusoidal2D(grid.shape)).flatten((Axis[PatchX], Axis[PatchY]))
      .relabel(Axis[PatchX |*| PatchY] -> Axis[Patch])

object ImageToPatchEmbedder:

  val patchSize = 16
  private val channels = 1

  /** @param embed The affine map from the flattened pixels of a patch to its embedding. */
  case class Params[PatchEmbedding, V](
      embed: AffineLayer.Params[PatchFeature, PatchEmbedding, V]
  )

  object Params:

    def xavierUniform[PatchEmbedding: Λ, V: IsFloating](embeddingExtent: AxisExtent[PatchEmbedding], key: Key, vtype: VType[V] = VType[Float32]): Params[PatchEmbedding, V] =
      Params(embed = AffineLayer.Params.xavierUniform(Axis[PatchFeature] -> patchSize * patchSize * channels, embeddingExtent, key, vtype))
