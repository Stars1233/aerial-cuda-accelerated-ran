#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Usage: ./delete_documents.sh "2024-08-25"

if [ -z "$1" ]; then
  echo "Error: No date provided."
  echo "Usage: $0 <date>"
  exit 1
fi

# Input date
input_date="$1"

if ! [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Error: Date format is incorrect. Use YYYY-MM-DD."
  exit 1
fi

datetime_str="${input_date} 00:00:00"
timestamp=$(date -d "$datetime_str" +%s)

token=$(python3 -c "import request_token;print(request_token.getToken());")

# Elasticsearch parameters
ES_HOST="https://gpuwa.nvidia.com/opensearch"
INDEX_NAME="df-sandbox-napostolakis-aerialopensearch-*"
DATE_FIELD="ts_cicd_datetime"
FIELD1="s_pipeline"
FIELD2="l_cicd_id"
FIELD3="s_test_name"

# Step 1: Find all tuples (field1, field2) where the datetime is past the input date
MATCH_QUERY=$(cat <<EOF
{
  "size": 10000,
  "_source": ["${FIELD1}", "${FIELD2}", "${FIELD3}"],
  "query": {
    "bool": {
      "must": [
        {"range": { "${DATE_FIELD}": { "lte": "${timestamp}" } }},
        {"term": {"s_type": "configuration_details" }}
      ]
    }
  }
}
EOF
)

# Execute the search request using curl to get the matching tuples
response=$(curl -s -X POST "${ES_HOST}/${INDEX_NAME}/_search" -H 'Content-Type: application/json' -H "Authorization: ${token}" -d"${MATCH_QUERY}")

# Extract the tuples (field1, field2) from the response
tuples=$(echo $response | jq -c '.hits.hits[]._source | {field1: .["'${FIELD1}'"], field2: .["'${FIELD2}'"], field3: .["'${FIELD3}'"]}' | sort | uniq)

for tuple in $tuples; do
  field1_value=$(echo $tuple | jq -r '.field1')
  field2_value=$(echo $tuple | jq -r '.field2')
  field3_value=$(echo $tuple | jq -r '.field3')

  DELETE_QUERY=$(cat <<EOF
  {
    "query": {
      "bool": {
        "must": [
          { "term": { "${FIELD1}": "${field1_value}" } },
          { "term": { "${FIELD2}": "${field2_value}" } },
          { "term": { "${FIELD3}": "${field3_value}" } }
        ]
      }
    }
  }
EOF
  )

  # Execute the delete request using curl
  curl -s -X POST "$ES_HOST/$INDEX_NAME/_delete_by_query" -H 'Content-Type: application/json' -H "Authorization: ${token}" -d"${DELETE_QUERY}"
done

echo "Documents successfully deleted."
