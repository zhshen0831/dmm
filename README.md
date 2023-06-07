This is an implementation for paper <u>DMM: Fast Map Matching for Cellular Data.</u>

---

### Overview
1. [Requirements](#requirements)
2. [Datasets](#Datasets)
3. [Execution](#execution)
4. [Parameter descriptions](#parameter)
5. [License](#license)
6. [Contact](#contact)

---

## 1. Requirements
The following modules are required.

* Ubuntu 16.04
* Python >=3.5 (`Anaconda3` recommended)
* PyTorch 0.4 (`virtualenv` recommended)
* Cuda 9.0

---

## 2. Datasets
In this dataset, we release the anonymized GPS data and corresponding cellular data collected from our volunteers for the evauluation. We have been negotiating with service providers to publish the cellular data safely. But due to security and privacy concerns, the cellular data used for training in the paper cannot be made public currently. The cellular data may be available from the service providers on a reasonable request. The data providers will review the applicants’ request and may provide them with limited access to some anonymous data.

* Training data
 >The dataset was collected by two major mobile carriers of a big city in China. Every time a subscriber of the carriers executes an action from a mobile device, a location sample is passively recorded by the carrier. The type of actions contains network service requests (calls, text messages and application Internet requests) and location updates (cell handover and periodic location update). The data include several fields, i.e., the anonymized subscriber identifier (Anonymized ID), the time information of the action (Time), the cell tower ID (CID) that the mobile device is attached during the action, the GPS coordinates of each cell tower.

 >Based on our testing, about 0.6 million anonymized cell tower sequences are required to train the model. For privacy concerns, it cannot be made public currently. You can apply for the access of the cellular dataset at `161.117.192.2:8080/index`. The data provider will review the applicants’ request and may provide them with limited access to some anonymous data. Dataset access will be only granted for approved legitimate researchers.

 Data format
 |Anonymized ID | Timestamp | LAC | CID | Latitude | Longtitude |
 |  :-----| :-----| ----: | :----: |:----: |:----: |
 |1B2A7| 08:08:19 |37146|19618|34.2482668| 108.9779304|


* Testing data
> Please refer to the `data` directory. We also collect the cell tower sequences in the real world for testing. Each sub-directory contains the data collected from one day. The following table describes the format of each record. The latitude and longtitude represent the GPS coordinates of each cell tower.

 Data format

 | Timestamp | Latitude | Longtitude |
 | :-----: | :----: | :----: |
 | 08:08:19 | 34.2482668 | 108.9779304 |

---
## 3. Execution

### 3.1 Training
```bash
$ python dmm.py -data="./data/train/" -hidden_size=128 -embedding_size=128 -save=100 -input_cell_size=9000 -output_road_size=20000 -epochs=10000 -batch=128
```

### 3.2 Testing
```bash
$ python dmm.py -data="./data/test/" -hidden_size=128 -embedding_size=128 -save=100 -input_cell_size=9000 -output_road_size=20000 -epochs=10000 -batch=128 -mode=1
```

---
## 4. Parameter descriptions
* `data` : Path to train and test data

* `checkpoint` : Path to saved checkpoint

* `pretrained_embedding` : Path to the pretrained cell embedding

* `num_layers` : The number of GRU layers

* `hidden_size` : The size of hidden state in the GRUs

* `embedding_size` : The size of cell embedding

* `print` : Print the results at every x iterations

* `dropout` : The dropout rate

* `learning_rate` : The learning rate

* `g_batch` : The maximum number of cells at each time

* `start_iteration` : The starting iteration

* `epochs` : The number of training epochs

* `save` : Save the model at every x iterations

* `cuda` : Whether to use GPU

* `criterion_name` : The optimizer to update the parameters

* `max_num_line` : The maximum lines to read

* `bidirectional` : Whether to use bidirectional rnn in the encoder

* `max_length` : The maximum length of the target road sequence

* `mode` : Train or test

* `batch` : The batch size

* `bucketsize` : The bucket size

* `input_cell_size` : The number of cell towers

* `output_road_size` : The number of road segments

---
## 5. License
The code is developed under the MPL-02.0 license.

---
## 6. Contact
If you have any questions or require further clarification, please do not hesitate to send an email to us (E-mail address： szh1095738849@stu.xjtu.edu.cn).
