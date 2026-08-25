package dataset

import dimwit.jax.Jax
import me.shadaj.scalapy.py

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

/** The Python this dataset reads through ScalaPy, wherever it is read from.
  *
  * Packaged as resources, a module is unpacked beside the others and imported from there; run
  * from the source tree, it is imported where it lies.
  */
private[dataset] object PythonModules:

  /** Touching `Jax.np` first makes sure DimWit has configured the interpreter and `sys.path`
    * before any Python object of ours is created.
    */
  def apply(name: String): py.Dynamic =
    Jax.np
    Option(getClass.getResourceAsStream(s"/python/$name.py")) match
      case Some(stream) =>
        try Files.copy(stream, unpacked.resolve(s"$name.py"), StandardCopyOption.REPLACE_EXISTING)
        finally stream.close()
      case None => fromSourceTree
    py.module(name)

  private lazy val unpacked: Path =
    val directory = Files.createTempDirectory("l-shape-python")
    py.module("sys").path.append(directory.toAbsolutePath.toString)
    Runtime.getRuntime.addShutdownHook(new Thread(() =>
      try Files.walk(directory).sorted(java.util.Comparator.reverseOrder()).forEach(Files.delete)
      catch case _: Exception => () // best effort cleanup
    ))
    directory

  private lazy val fromSourceTree: Unit = py.module("sys").path.append("./dataset/src/main/resources/python")
