package dataset

import dimwit.jax.Jax
import me.shadaj.scalapy.py

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

/** The Python a dataset reads through ScalaPy, wherever it is read from.
  *
  * Packaged as resources, a module is unpacked beside the others and imported from there; run
  * from the source tree, it is imported where it lies.
  */
object PythonModules:

  /** Touching `Jax.np` first makes sure DimWit has configured the interpreter and `sys.path`
    * before any Python object of ours is created.
    *
    * @param sourceTree Where the module lies in the source tree, for a run from it.
    */
  def apply(name: String, sourceTree: String = "./dataset/src/main/resources/python"): py.Dynamic =
    Jax.np
    Option(getClass.getResourceAsStream(s"/python/$name.py")) match
      case Some(stream) =>
        try Files.copy(stream, unpacked.resolve(s"$name.py"), StandardCopyOption.REPLACE_EXISTING)
        finally stream.close()
      case None =>
        val path = py.module("sys").path
        if !path.as[Seq[String]].contains(sourceTree) then path.append(sourceTree)
    py.module(name)

  private lazy val unpacked: Path =
    val directory = Files.createTempDirectory("l-shape-python")
    py.module("sys").path.append(directory.toAbsolutePath.toString)
    Runtime.getRuntime.addShutdownHook(new Thread(() =>
      try Files.walk(directory).sorted(java.util.Comparator.reverseOrder()).forEach(Files.delete)
      catch case _: Exception => () // best effort cleanup
    ))
    directory
