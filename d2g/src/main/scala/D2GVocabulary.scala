package d2g

import dimwit.*

export d2s.{Width, Height, Channel, Patch, Node, Pixel, Embedding, Pixels}

// Edges and their properties
trait Edge derives Label // A record edge
trait LinkedNode derives Label // The node a relationship links to
