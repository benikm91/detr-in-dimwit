package common

import dimwit.*
import deepwit.base.AffineLayer
import deepwit.embedder.PositionalEncoding.sinusoidal2D
import dimwit.Label as Λ

trait PatchFeature derives Label

/** Cuts a drawing into non-overlapping 16x16 patches and embeds every patch into a sequence
  * element.
  *
  * Each patch is flattened and embedded by one affine map, and the patch grid is enriched with a 2D
  * sinusoidal positional encoding before being flattened into a sequence. This is exactly the
  * convolution with stride equal to kernel that patch embeddings usually go by, but computed as a
  * reshape and a matrix product, so it runs wherever those do.
  *
  * @param width The axis of the drawing width; the drawing must cut into whole patches.
  * @param height The axis of the drawing height; likewise.
  * @param channel The axis of the single drawing channel.
  * @param params The learnable parameters.
  */
class ImageToPatchEmbedder[Width: Λ, Height: Λ, Channel: Λ, PatchEmbedding: Λ, V: IsFloating](
    width: Axis[Width],
    height: Axis[Height],
    channel: Axis[Channel],
    params: ImageToPatchEmbedder.Params[PatchEmbedding, V]
) extends (Tensor3[Width, Height, Channel, V] => Tensor2[Width |*| Height, PatchEmbedding, V]):

  import ImageToPatchEmbedder.patchSize

  private val embed = AffineLayer(params.embed)

  override def apply(img: Tensor3[Width, Height, Channel, V]): Tensor2[Width |*| Height, PatchEmbedding, V] =
    trait PatchX derives Label
    trait PatchY derives Label
    val patches = img
      .unflatten(width, Shape(Axis[PatchX] -> img.shape(width) / patchSize, width -> patchSize))
      .unflatten(height, Shape(Axis[PatchY] -> img.shape(height) / patchSize, height -> patchSize))
    val grid = patches
      .vmap(Axis[PatchX])(_.vmap(Axis[PatchY])(p => embed(p.flatten.relabelTo(Axis[PatchFeature]))))
    (grid + sinusoidal2D(grid.shape)).flatten((Axis[PatchX], Axis[PatchY]))
      .relabel(Axis[PatchX |*| PatchY] -> Axis[Width |*| Height])

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
