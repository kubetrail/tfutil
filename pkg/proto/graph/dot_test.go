package graph

import (
	"bytes"
	"testing"

	"google.golang.org/protobuf/encoding/prototext"
)

// tfLinalgNormGraph is a graph representation of tf.linalg.norm func
var tfLinalgNormGraph = `
node {
  name: "tensor"
  op: "Placeholder"
  attr {
    key: "_user_specified_name"
    value {
      s: "tensor"
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_DOUBLE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
      }
    }
  }
}
node {
  name: "norm/mul"
  op: "Mul"
  input: "tensor"
  input: "tensor"
  attr {
    key: "T"
    value {
      type: DT_DOUBLE
    }
  }
}
node {
  name: "norm/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "norm/Sum"
  op: "Sum"
  input: "norm/mul"
  input: "norm/Const"
  attr {
    key: "T"
    value {
      type: DT_DOUBLE
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: true
    }
  }
}
node {
  name: "norm/Sqrt"
  op: "Sqrt"
  input: "norm/Sum"
  attr {
    key: "T"
    value {
      type: DT_DOUBLE
    }
  }
}
node {
  name: "norm/Squeeze"
  op: "Squeeze"
  input: "norm/Sqrt"
  attr {
    key: "T"
    value {
      type: DT_DOUBLE
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
      }
    }
  }
}
node {
  name: "Identity"
  op: "Identity"
  input: "norm/Squeeze"
  attr {
    key: "T"
    value {
      type: DT_DOUBLE
    }
  }
}
versions {
  producer: 808
}
`

var expectedDotOutput = `"tensor" [fillcolor="antiquewhite3", style="filled", label="tensor"]
"norm/mul" [label="norm/mul"]
"tensor" -> "norm/mul"
"tensor" -> "norm/mul"
"norm/Const" [fillcolor="aquamarine3", style="filled", label="norm/Const"]
"norm/Sum" [label="norm/Sum"]
"norm/mul" -> "norm/Sum"
"norm/Const" -> "norm/Sum"
"norm/Sqrt" [label="norm/Sqrt"]
"norm/Sum" -> "norm/Sqrt"
"norm/Squeeze" [label="norm/Squeeze"]
"norm/Sqrt" -> "norm/Squeeze"
"Identity" [label="Identity"]
"norm/Squeeze" -> "Identity"
`

// TestDef_PrintDotNotation tests output of printing dot notation
func TestDef_PrintDotNotation(t *testing.T) {
	graphDef, err := NewGraphDef()
	if err != nil {
		t.Fatal(err)
	}

	if err := prototext.Unmarshal([]byte(tfLinalgNormGraph), graphDef); err != nil {
		t.Fatal(err)
	}

	b, err := graphDef.PrintDotNotation()
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(b, []byte(expectedDotOutput)) {
		t.Fatal("dot notion output does not match expected output")
	}
}
