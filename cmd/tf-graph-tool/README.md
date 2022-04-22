# tf-graph-tool
This is a CLI to parse TensorFlow graphs and perform operations on
graph nodes and tensors.

## disclaimer
>The use of this tool does not guarantee security or suitability
for any particular use. Please review the code and use at your own risk.

## installation
This step assumes you have [Go compiler toolchain](https://go.dev/dl/)
installed on your system.

First make sure you have downloaded and installed `TensorFlow`
[C-library](https://www.tensorflow.org/install/lang_c) and make
sure you are able to build and run the "hello-world" as
described on that page.

Finally, install `tf-graph-tool`:
```bash
go install github.com/kubetrail/tfutil/tree/main/cmd/tf-graph-tool@latest
```

Optionally, install shell completion. For instance, `bash` completion can be installed
by adding following line to your `.bashrc`:
```bash
source <(mksecret completion bash)
```

## download an example graph
Download and extract `.pb` file from the
[tar bundle](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz), 
which is a pretrained image labeling graph.

## decode graph
Graph can be decoded to `JSON` format
```bash
tf-graph-tool graph decode inception_v3_2016_08_28_frozen.pb
```

output below has been redacted for brevity:
```json
{
  "node": [
    {
      "name": "input",
      "op": "Placeholder",
      "attr": {
        "dtype": {
          "Value": {
            "Type": 1
          }
        },
        "shape": {
          "Value": {
            "Shape": {
              "dim": [
                {
                  "size": 1
                },
                {
                  "size": 299
                },
                {
                  "size": 299
                },
                {
                  "size": 3
                }
              ]
            }
          }
        }
      }
    }
  ]
}
```

Similarly, output formats can be changed to proto text:
```bash
tf-graph-tool graph decode --output-format=pbtxt inception_v3_2016_08_28_frozen.pb
```
which produces output as shown below (redacted again)
```text
node:{name:"input" op:"Placeholder" attr:{key:"dtype" value:{type:DT_FLOAT}} attr:{key:"shape" value:{shape:{dim:{size:1} dim:{size:299} dim:{size
:299} dim:{size:3}}}}} node:{name:"InceptionV3/Conv2d_1a_3x3/weights" op:"Const" attr:{key:"dtype" value:{type:DT_FLOAT}} attr:{key:"value" value:
{tensor:{dtype:DT_FLOAT tensor_shape:{dim:{size:3} dim:{size:3} dim:{size:3} dim:{size:32}} tensor_content:...
```

## node operations
### list
Nodes can be listed using a `selector`. For instance, `Placeholder` nodes
can be listed as shown below:
```bash
tf-graph-tool node list --selector=op=placeholder inception_v3_2016_08_28_frozen.pb
```
```json
[
  "input"
]
```

Similarly, `constant` nodes can be identified:
```bash
tf-graph-tool node list --selector=op=constant inception_v3_2016_08_28_frozen.pb
```
Output below has been redacted to show here
```json
[
  "InceptionV3/Conv2d_1a_3x3/weights",
  "InceptionV3/Conv2d_1a_3x3/BatchNorm/beta",
  "InceptionV3/Conv2d_1a_3x3/BatchNorm/moving_mean",
  "InceptionV3/Conv2d_1a_3x3/BatchNorm/moving_variance",
  "InceptionV3/InceptionV3/Conv2d_1a_3x3/BatchNorm/batchnorm/add/y",
  "InceptionV3/Conv2d_2a_3x3/weights",
  "InceptionV3/Conv2d_2a_3x3/BatchNorm/beta",
  "InceptionV3/Conv2d_2a_3x3/BatchNorm/moving_mean",
  "InceptionV3/Conv2d_2a_3x3/BatchNorm/moving_variance"
]
```

### node details
Detailed info about the nodes can be obtained using `get` command:
```bash
tf-graph-tool node get --name=input inception_v3_2016_08_28_frozen.pb
```
```json
{
  "node": [
    {
      "name": "input",
      "op": "Placeholder",
      "attr": {
        "dtype": {
          "Value": {
            "Type": 1
          }
        },
        "shape": {
          "Value": {
            "Shape": {
              "dim": [
                {
                  "size": 1
                },
                {
                  "size": 299
                },
                {
                  "size": 299
                },
                {
                  "size": 3
                }
              ]
            }
          }
        }
      }
    }
  ],
  "versions": {
    "producer": 987
  },
  "library": {}
}
```

This allows us to identify that the input to the graph is a tensor of shape
`[1, 299, 299, 3]` and its data type as `1`, which is `float32`

### run a node
Tensors embedded in const nodes can be extracted:
```bash
tf-graph-tool node run --name=InceptionV3/Predictions/Shape inception_v3_2016_08_28_frozen.pb 2>/dev/null | jq '.'
```
```json
{
  "type": "tensor",
  "tfDataType": "Int32",
  "goDataType": "int32",
  "shape": [
    2
  ],
  "value": [
    1,
    1001
  ]
}
```

## tensor operations
Graphs can be composed by embedding tensor. First prepare a file with serialized tensor in it:
```bash
cat /tmp/tensor.json
```
```json
{
  "type": "tensor",
  "tfDataType": "Int32",
  "goDataType": "int32",
  "shape": [
    2
  ],
  "value": [
    1,
    1001
  ]
}
```

Embed this tensor as a new node with name `testNode`
```bash
tf-graph-tool tensor encode --name=testNode /tmp/tensor.json
```
```json
{
  "node": [
    {
      "name": "testNode",
      "op": "Const",
      "attr": {
        "dtype": {
          "Value": {
            "Type": 3
          }
        },
        "value": {
          "Value": {
            "Tensor": {
              "dtype": 3,
              "tensor_shape": {
                "dim": [
                  {
                    "size": 2
                  }
                ]
              },
              "tensor_content": "AQAAAOkDAAA="
            }
          }
        }
      }
    }
  ],
  "versions": {
    "producer": 987
  },
  "library": {}
}
```

## merge graphs
Assuming you have tensor from previous example in a file `/tmp/tensor.json`, let's create
two graphs:
```bash
tf-graph-tool tensor encode --name=testNode1 --output-format=pb /tmp/tensor.json > /tmp/graph1.pb
tf-graph-tool tensor encode --name=testNode2 --output-format=pb /tmp/tensor.json > /tmp/graph2.pb
```

These graphs can be merged:
```bash
tf-graph-tool graph merge /tmp/graph1.pb /tmp/graph2.pb
```
```json
{
  "node": [
    {
      "name": "testNode1",
      "op": "Const",
      "attr": {
        "dtype": {
          "Value": {
            "Type": 3
          }
        },
        "value": {
          "Value": {
            "Tensor": {
              "dtype": 3,
              "tensor_shape": {
                "dim": [
                  {
                    "size": 2
                  }
                ]
              },
              "tensor_content": "AQAAAOkDAAA="
            }
          }
        }
      }
    },
    {
      "name": "testNode2",
      "op": "Const",
      "attr": {
        "dtype": {
          "Value": {
            "Type": 3
          }
        },
        "value": {
          "Value": {
            "Tensor": {
              "dtype": 3,
              "tensor_shape": {
                "dim": [
                  {
                    "size": 2
                  }
                ]
              },
              "tensor_content": "AQAAAOkDAAA="
            }
          }
        }
      }
    }
  ],
  "versions": {
    "producer": 987
  },
  "library": {}
}
```
