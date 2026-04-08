#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import yaml
import h5py
import numpy as np
from typing import List, Tuple
import os
from fapi_types import (
	CellConfig, PDU, IND, CellSlotConfig, SlotConfig,
	TvChannelType, IndicationType, PuschReq, PuschInd, SFN_iter
)
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 400)

np.set_printoptions(threshold=20)  # Control the number of elements to display
np.set_printoptions(edgeitems=20)  # Control the number of edge items to display
np.set_printoptions(linewidth=200) # Control the width of the display
import clickhouse_connect

def create_fapi_dataframe(h5_file: h5py.File, nUEs: int, slot: float, cell_id: int) -> pd.DataFrame:
	"""Create a DataFrame with FapiRecord columns from PDUs and INDs in the HDF5 file.

	Args:
		h5_file: Open HDF5 file object
		nUEs: Total number of PUSCH PDUs across all cells
		slot: Slot number for this DataFrame
		cell_id: Physical cell ID for this cell
	"""
	records = []

	pusch_num = 1
	while f'PDU{pusch_num}' in h5_file:
		try:
			pdu_data = h5_file[f'PDU{pusch_num}'][0]
			if int(pdu_data['type']) != TvChannelType.TV_PUSCH:
				pusch_num += 1
				continue

			pduIndIdx = int(pdu_data['idxInd'])
			ind_data = h5_file[f'IND{pduIndIdx}'][0]
			indPduIdx = int(ind_data['idxPdu'])
			if pusch_num != indPduIdx:
				print(f"PDU{pusch_num}->idxInd:{indPduIdx} and IND{pduIndIdx}->pduIdx:{indPduIdx} indices mismatch: {pusch_num} != {indPduIdx}")
				exit(1)
			pdu_payload = h5_file[f'PDU{pusch_num}_payload'][()].tolist()
		
			record = {
				'TsTaiNs': None,
				'TsSwNs': None,
				'SFN': None,
				'Slot': slot,
				'nUEs': nUEs,
				'pduBitmap': pdu_data['pduBitmap'],
				'rnti': int(pdu_data['RNTI']),
				'BWPSize': int(pdu_data['BWPSize']),
				'BWPStart': int(pdu_data['BWPStart']),
				'SubcarrierSpacing': int(pdu_data['SubcarrierSpacing']),
				'CyclicPrefix': int(pdu_data['CyclicPrefix']),
				'targetCodeRate': int(pdu_data['targetCodeRate']),
				'qamModOrder': int(pdu_data['qamModOrder']),
				'mcsIndex': int(pdu_data['mcsIndex']),
				'mcsTable': int(pdu_data['mcsTable']),
				'TransformPrecoding': int(pdu_data['TransformPrecoding']),
				'dataScramblingId': int(pdu_data['dataScramblingId']),
				'nrOfLayers': int(pdu_data['nrOfLayers']),
				'ulDmrsSymbPos': int(pdu_data['DmrsSymbPos']),
				'dmrsConfigType': int(pdu_data['dmrsConfigType']),
				'ulDmrsScramblingId': int(pdu_data['DmrsScramblingId']),
				'puschIdentity': int(pdu_data['puschIdentity']),
				'SCID': int(pdu_data['SCID']),
				'numDmrsCdmGrpsNoData': int(pdu_data['numDmrsCdmGrpsNoData']),
				'dmrsPorts': int(pdu_data['dmrsPorts']),
				'resourceAlloc': int(pdu_data['resourceAlloc']),
				'rbStart': int(pdu_data['rbStart']),
				'rbSize': int(pdu_data['rbSize']),
				'VRBtoPRBMapping': int(pdu_data['VRBtoPRBMapping']),
				'FrequencyHopping': int(pdu_data['FrequencyHopping']),
				'txDirectCurrentLocation': int(pdu_data['txDirectCurrentLocation']),
				'uplinkFrequencyShift7p5khz': int(pdu_data['uplinkFrequencyShift7p5khz']),
				'StartSymbolIndex': int(pdu_data['StartSymbolIndex']),
				'NrOfSymbols': int(pdu_data['NrOfSymbols']),
				'rvIndex': int(pdu_data['rvIndex']),
				'harqProcessID': int(pdu_data['harqProcessID']),
				'newDataIndicator': int(pdu_data['newDataIndicator']),
				'TBSize': int(pdu_data['TBSize']),
				'numCb': int(pdu_data['numCb']),
				'numPRGs': int(pdu_data['numPRGs']),
				'prgSize': int(pdu_data['prgSize']),
				'tbCrcStatus': int(ind_data['TbCrcStatus']),
				'CQI': np.min([40, np.float32(pdu_data['sinrdB'])]),
				'timingAdvance': np.float32(ind_data['TimingAdvanceNano'])/1e9,
				'rssi': np.float32(pdu_data['dmrsRssiReportedDb']),
				'pduLen': len(pdu_payload[0]),
				'pduData': pdu_payload[0],
				'CellId': cell_id
			}
			records.append(record)
		except (KeyError, ValueError) as e:
			print(f"Error processing PDU{pusch_num} and IND{pusch_num}: {str(e)}")
			print(f"Available fields in PDU{pusch_num} and IND{pusch_num}: {pdu_data.dtype.names}")
			raise
		pusch_num += 1
	# Convert to DataFrame
	df = pd.DataFrame(records)
	return df

def get_clickhouse_stats():
	"""Connect to ClickHouse and get statistics for fapi and fh tables."""
	client = clickhouse_connect.get_client(host='localhost')

	# Get comprehensive table statistics
	result = client.query('''
		SELECT
			name AS table,
			round(total_bytes / (1024 * 1024), 2) AS size_in_megabytes,
			total_rows AS number_of_rows
		FROM system.tables
		WHERE database = 'default' AND name IN ('fapi', 'fapi_disk', 'fh', 'tv_data')
	''').result_set

	print("\nClickHouse Table Statistics:")

	# Calculate column widths
	table_width = max(len("table"), max(len(row[0]) for row in result))
	size_width = max(len("size_in_megabytes"), max(len(f"{row[1]:,.2f}") for row in result))
	rows_width = max(len("number_of_rows"), max(len(f"{row[2]:,}") for row in result))

	# Print header
	print(f"┌─{'─' * (table_width + 3)}─┬─{'─' * size_width}─┬─{'─' * rows_width}─┐")
	print(f"│ {'table':<{table_width + 3}} │ {'size_in_megabytes':<{size_width}} │ {'number_of_rows':<{rows_width}} │")
	print(f"├─{'─' * (table_width + 3)}─┼─{'─' * size_width}─┼─{'─' * rows_width}─┤")

	# Print rows
	for i, (table_name, size_mb, rows) in enumerate(result, 1):
		print(f"│ {i}. {table_name:<{table_width}} │ {size_mb:>{size_width},.2f} │ {rows:>{rows_width},} │")

	# Print footer
	print(f"└─{'─' * (table_width + 3)}─┴─{'─' * size_width}─┴─{'─' * rows_width}─┘")

def read_yaml_and_h5(CUBB_HOME: str, yaml_file: str) -> Tuple[List[CellConfig], List[SlotConfig]]:
	tv_dir = os.path.join(CUBB_HOME, 'testVectors')
	yaml_path = os.path.join(tv_dir, 'multi-cell', yaml_file)
	print(f"Reading YAML file: {yaml_path}")

	with open(yaml_path, 'r') as f:
		yaml_data = yaml.safe_load(f)

	# Get the parent directory of the YAML file for HDF5 files

	# First pass: read all cell configs
	cell_configs = [
		CellConfig(**{name: cell_config[name].item() for name in CellConfig.__dataclass_fields__.keys()})
		for cell_data in yaml_data['Cell_Configs']
		for cell_config in [h5py.File(os.path.join(tv_dir, cell_data), 'r')['Cell_Config'][0]]
	]

	# Workaround for same phyCellId in Cell_Config TV of different cells
	if len(cell_configs) > 1:
		# Get phyCellIds from first two cells
		phyCellId_0 = cell_configs[0].phyCellId
		phyCellId_1 = cell_configs[1].phyCellId
	
		if phyCellId_0 == phyCellId_1:
			# Update phyCellIds by adding cell index
			for idx, config in enumerate(cell_configs):
				config.phyCellId += idx
				print(f"Updated phyCellId: cell_id={idx} to phyCellId={config.phyCellId}")

	# Process slots from SCHED section
	slot_configs = []
	harq_process_id = 0  # Start with 0 for first slot

	for slot_data in yaml_data['SCHED']:
		slot = slot_data['slot']
		total_pusch_pdus = 0
		active_ul_slots = 0
		dfs = []
		x_tfs = []
		drop_slot = True
		if slot == 20:
			print(f"Skipping slots higher than 20, because mu=1 stops at 20.")
			print(f"If TVs in slots > 20 differ from those in lower slots there will be mismatches")
			break
	
		# Process each channel in the slot
		for cell_idx, cell_data in enumerate(slot_data['config']):
			for channel in cell_data['channels']:
				full_channel_path = os.path.join(tv_dir, channel)
				with h5py.File(full_channel_path, 'r') as f:
					# Count PUSCH PDUs
					pdu_num = 1
					while f'PDU{pdu_num}' in f:
						pdu_data = f[f'PDU{pdu_num}'][0]
						if int(pdu_data['type']) == TvChannelType.TV_PUSCH:
							total_pusch_pdus += 1
							drop_slot = False
						pdu_num += 1
		if drop_slot:
			# Remove this slot from yaml_data['SCHED']
			yaml_data['SCHED'].remove(slot_data)
			continue
	
		# Process each channel in the slot
		for cell_idx, cell_data in enumerate(slot_data['config']):
			for channel in cell_data['channels']:
				full_channel_path = os.path.join(tv_dir, channel)
				with h5py.File(full_channel_path, 'r') as f:
					df = create_fapi_dataframe(f, total_pusch_pdus, slot, cell_configs[cell_idx].phyCellId)
					if not df.empty:
						# Update HARQ process IDs for this slot
						df['harqProcessID'] = harq_process_id
						dfs.append(df)
				
					# Read X_tf for this cell
					if 'X_tf' in f:
						x_tfs.append(f['X_tf'][:])
	
		harq_process_id = (harq_process_id + 1) % 16
		# Create slot config
		slot_configs.append(SlotConfig(
			slot=slot,
			total_pusch_pdus=total_pusch_pdus,
			dataframe=pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(),
			x_tf=x_tfs if x_tfs else None
		))

	if not slot_configs:
		print("No slots with PUSCH PDUs found in the test vectors")
		exit(1)

	return cell_configs, slot_configs

def drop_startup_rows(slot_configs: List[SlotConfig]):
	client = clickhouse_connect.get_client(host='localhost')

	# Query 100 rows sorted by TsTaiNs, SFN, Slot
	df = client.query_df('SELECT * FROM fapi ORDER BY TsTaiNs, SFN, Slot, CellId LIMIT 100')

	if df.empty:
		print("No rows found in database")
		exit(1)

	# Find first row with non-zero pduLen
	first_row_idx = next((idx for idx, row in df.iterrows() if row['pduLen'] > 0), None)

	if first_row_idx is None:
		print("No rows found with non-zero pduLen in first 1000 rows of database")
		exit(1)

	first_row = df.iloc[first_row_idx]

	# Delete startup rows from database and DataFrame
	if first_row_idx > 0:
		print(f"Deleting first {first_row_idx} rows from before timestamp: {first_row['TsTaiNs']}")
		fapi_query = f"ALTER TABLE fapi DELETE WHERE TsTaiNs < toDateTime64(\'{first_row['TsTaiNs']}\',9)"
		fh_query = f"ALTER TABLE fh DELETE WHERE TsTaiNs < toDateTime64(\'{first_row['TsTaiNs']}\',9)"
		client.query(fapi_query)
		client.query(fh_query)

def setup_columns(slot_configs: List[SlotConfig]):
	columns_to_drop = [
		'pduBitmap', 'resourceAlloc', 'BWPStart', 'BWPSize',
		'TsSwNs', 'FrequencyHopping', 'digBFInterface', 'rbBitmap',
		'VRBtoPRBMapping', 'uplinkFrequencyShift7p5khz', 'TransformPrecoding',
		'txDirectCurrentLocation', 'numPRGs', 'prgSize'
	]
	print(f"Not comparing columns: {columns_to_drop}")

	# Rename columns in slot DataFrames and drop unwanted columns
	for slot in slot_configs:
		if not slot.dataframe.empty:
			slot.dataframe = slot.dataframe.rename(columns={'tbCrcStatus': 'tbCrcFail'})
			slot.dataframe = slot.dataframe.drop(columns=columns_to_drop, errors='ignore')


def compare_tv_to_fapi(slot_configs: List[SlotConfig], printMismatches: bool):
	if printMismatches is not False:
		printMismatch = True
	else:
		printMismatch = False

	client = clickhouse_connect.get_client(host='localhost')
	retCode = 0
	# Create table for test vector data
	create_table_query = """
	CREATE TABLE IF NOT EXISTS tv_data (
		Slot UInt16,
		CellId UInt16,
		nUEs UInt16,
		rnti UInt16,
		SubcarrierSpacing UInt8,
		CyclicPrefix UInt8,
		targetCodeRate UInt16,
		qamModOrder UInt8,
		mcsIndex UInt8,
		mcsTable UInt8,
		dataScramblingId UInt16,
		nrOfLayers UInt8,
		ulDmrsSymbPos UInt16,
		dmrsConfigType UInt8,
		ulDmrsScramblingId UInt16,
		puschIdentity UInt16,
		SCID UInt8,
		numDmrsCdmGrpsNoData UInt8,
		dmrsPorts UInt16,
		rbStart UInt16,
		rbSize UInt16,
		StartSymbolIndex UInt8,
		NrOfSymbols UInt8,
		rvIndex UInt8,
		harqProcessID UInt8,
		newDataIndicator UInt8,
		TBSize UInt32,
		numCb UInt8,
		tbCrcFail UInt8,
		CQI Float32,
		timingAdvance Float32,
		rssi Float32,
		pduLen UInt32,
		pduData Array(UInt8)
	) ENGINE = Memory
	"""

	# Drop existing table if it exists
	client.query('DROP TABLE IF EXISTS tv_data')
	client.query(create_table_query)

	# Insert test vector data
	for slot in slot_configs:
		# Select only the columns that match our table schema
		columns_to_keep = [
			'Slot', 'CellId', 'nUEs', 'rnti', 'SubcarrierSpacing', 'CyclicPrefix',
			'targetCodeRate', 'qamModOrder', 'mcsIndex', 'mcsTable', 'dataScramblingId',
			'nrOfLayers', 'ulDmrsSymbPos', 'dmrsConfigType', 'ulDmrsScramblingId',
			'puschIdentity', 'SCID', 'numDmrsCdmGrpsNoData', 'dmrsPorts', 'rbStart',
			'rbSize', 'StartSymbolIndex', 'NrOfSymbols', 'rvIndex', 'harqProcessID',
			'newDataIndicator', 'TBSize', 'numCb', 'tbCrcFail', 'CQI', 'timingAdvance',
			'rssi', 'pduLen', 'pduData'
		]
	
		df = slot.dataframe[columns_to_keep].copy()
	
		# Convert DataFrame columns to correct types
		uint32_cols = ['TBSize', 'pduLen']
		uint16_cols = ['Slot', 'CellId', 'nUEs', 'rnti', 'targetCodeRate', 'dataScramblingId',
							 'ulDmrsSymbPos', 'ulDmrsScramblingId', 'puschIdentity', 'dmrsPorts',
							 'rbStart', 'rbSize']
		uint8_cols = ['SubcarrierSpacing', 'CyclicPrefix', 'qamModOrder', 'mcsIndex',
							'mcsTable', 'nrOfLayers', 'dmrsConfigType', 'SCID',
							'numDmrsCdmGrpsNoData', 'StartSymbolIndex', 'NrOfSymbols',
							'rvIndex', 'harqProcessID', 'newDataIndicator', 'numCb',
							'tbCrcFail']
		float32_cols = ['CQI', 'timingAdvance', 'rssi']
			
		for col in float32_cols:
			df[col] = df[col].astype('float64')
		for col in uint32_cols:
			df[col] = df[col].astype('uint32')
		for col in uint16_cols:
			df[col] = df[col].astype('uint16')
		for col in uint8_cols:
			df[col] = df[col].astype('uint8')
			
		# Insert DataFrame with correct types
		client.insert_df('tv_data', df)


	comparison_query = """
	WITH mismatches AS (
		SELECT
			tv.Slot,
			tv.CellId,
			countIf(f.harqProcessID != tv.harqProcessID) as harqProcessID_mismatch,
			countIf(f.SubcarrierSpacing != tv.SubcarrierSpacing) as SubcarrierSpacing_mismatch,
			countIf(f.CyclicPrefix != tv.CyclicPrefix) as CyclicPrefix_mismatch,
			countIf(f.targetCodeRate != tv.targetCodeRate) as targetCodeRate_mismatch,
			countIf(f.qamModOrder != tv.qamModOrder) as qamModOrder_mismatch,
			countIf(f.mcsIndex != tv.mcsIndex) as mcsIndex_mismatch,
			countIf(f.mcsTable != tv.mcsTable) as mcsTable_mismatch,
			countIf(f.dataScramblingId != tv.dataScramblingId) as dataScramblingId_mismatch,
			countIf(f.nrOfLayers != tv.nrOfLayers) as nrOfLayers_mismatch,
			countIf(f.ulDmrsSymbPos != tv.ulDmrsSymbPos) as ulDmrsSymbPos_mismatch,
			countIf(f.dmrsConfigType != tv.dmrsConfigType) as dmrsConfigType_mismatch,
			countIf(f.ulDmrsScramblingId != tv.ulDmrsScramblingId) as ulDmrsScramblingId_mismatch,
			countIf(f.puschIdentity != tv.puschIdentity) as puschIdentity_mismatch,
			countIf(f.SCID != tv.SCID) as SCID_mismatch,
			countIf(f.numDmrsCdmGrpsNoData != tv.numDmrsCdmGrpsNoData) as numDmrsCdmGrpsNoData_mismatch,
			countIf(f.dmrsPorts != tv.dmrsPorts) as dmrsPorts_mismatch,
			countIf(f.rbStart != tv.rbStart) as rbStart_mismatch,
			countIf(f.rbSize != tv.rbSize) as rbSize_mismatch,
			countIf(f.StartSymbolIndex != tv.StartSymbolIndex) as StartSymbolIndex_mismatch,
			countIf(f.NrOfSymbols != tv.NrOfSymbols) as NrOfSymbols_mismatch,
			countIf(f.rvIndex != tv.rvIndex) as rvIndex_mismatch,
			countIf(f.newDataIndicator != tv.newDataIndicator) as newDataIndicator_mismatch,
			countIf(f.TBSize != tv.TBSize) as TBSize_mismatch,
			countIf(f.numCb != tv.numCb) as numCb_mismatch,
			countIf(f.tbCrcFail != tv.tbCrcFail) as tbCrcFail_mismatch,
			countIf(abs(f.CQI - tv.CQI) > 1) as CQI_mismatch,
			countIf(abs(f.timingAdvance - tv.timingAdvance) > 0.001) as timingAdvance_mismatch,
			countIf(abs(f.rssi - tv.rssi) > 1) as rssi_mismatch
		FROM tv_data tv
		JOIN fapi f ON tv.Slot = f.Slot AND tv.CellId = f.CellId AND tv.rnti = f.rnti
		GROUP BY tv.Slot, tv.CellId, tv.harqProcessID
	)
	SELECT * FROM mismatches
	ORDER BY Slot, CellId
	"""
	comparison_df = client.query_df(comparison_query)
	# Filter columns to show only those with non-zero mismatches
	mismatch_cols = [col for col in comparison_df.columns if col.endswith('_mismatch')]
	non_zero_cols = ['Slot', 'CellId'] # Always show these columns
	#Change this to True to show that harqID doesn't match, and we expect that.
	for col in mismatch_cols:
		if comparison_df[col].sum() > 0:
			non_zero_cols.append(col)
			# Not fixing harq because it's a good canary to show that the others work
			if col != 'harqProcessID_mismatch':
				printMismatch = True
				retCode = 1
	if printMismatch:
		print("\nMismatch summary by column:")
		print(comparison_df[non_zero_cols])
	else:
		print("No fapi mismatches found")

	return retCode

def compare_tv_to_fh(slot_configs: List[SlotConfig]):
	retCode = 0
	matches = []
	mismatches = []
	magicScaleFactor = 1 # This works for CICD fs_offset_ul: 0 and exponent_ul: 4
	xtfThreshold = 0.05
	client = clickhouse_connect.get_client(host='localhost')

	# First check each slot config against its first FH entry
	for slot_config in slot_configs:
		slot = slot_config.slot
		for cell_idx, cell_config in enumerate(cell_configs):
			cell_id = cell_config.phyCellId

			# Get first FH entry for this slot/cell
			fh_query = f""" SELECT TsTaiNs, SFN, Slot, CellId, fhData FROM fh
			WHERE Slot = {slot} AND CellId = {cell_id} ORDER BY TsTaiNs, SFN, Slot, CellId LIMIT 1
			"""
			fh_row = client.query_df(fh_query)
		
			if fh_row.empty:
				print(f"No FH data found for Slot {slot}, CellId {cell_id}")
				continue
			
			# Process FH data
			fh_samp = (np.array(fh_row.iloc[0]['fhData'], dtype=np.int16).view(np.float16)).astype(np.float32)
			fh_x_tf = fh_samp.reshape(4, 14, 273 * 12*2)*magicScaleFactor
			
			# Get TV x_tf
			tv_x_tf = slot_config.x_tf[cell_idx].view(np.float32)
			
			# Compare shapes and values
			if fh_x_tf.shape == tv_x_tf.shape:
				if np.allclose(fh_x_tf, tv_x_tf, rtol=1, atol=xtfThreshold):
					matches.append((slot, cell_id))
				
					# Now verify all other FH entries for this slot/cell have the same data
					verify_query = f""" WITH first_entry AS (
						SELECT fhData FROM fh WHERE Slot = {slot} AND CellId = {cell_id}
						ORDER BY TsTaiNs, SFN, Slot, CellId LIMIT 1
					)
					SELECT count(*) as total_rows,
						   countIf(fhData = (SELECT fhData FROM first_entry)) as matching_rows
					FROM fh
					WHERE Slot = {slot} AND CellId = {cell_id}
					"""
					verify_result = client.query_df(verify_query)
				
					if not verify_result.empty:
						total_rows = verify_result.iloc[0]['total_rows']
						matching_rows = verify_result.iloc[0]['matching_rows']
						print(f"Checked Slot {slot}, CellId {cell_id}, {total_rows} rows, {matching_rows} matched")
						if total_rows != matching_rows:
							print(f"\nWarning: Not all FH entries match for Slot {slot}, CellId {cell_id}")
							print(f"Total rows: {total_rows}, Matching rows: {matching_rows}")
							print(f"First 10 values of reference fhData: {[f'{x:.7f}' for x in fh_x_tf[0,2,0:10]]}")
						
							# Get a sample of mismatching rows
							sample_query = f"""
							WITH first_entry AS (
								SELECT fhData
								FROM fh
								WHERE Slot = {slot} AND CellId = {cell_id}
								ORDER BY TsTaiNs, SFN, Slot, CellId
								LIMIT 1
							)
							SELECT TsTaiNs, SFN, Slot, CellId
							FROM fh
							WHERE Slot = {slot}
							  AND CellId = {cell_id}
							  AND fhData != (SELECT fhData FROM first_entry)
							LIMIT 5
							"""
							sample_mismatches = client.query_df(sample_query)
							print("\nSample of mismatching entries:")
							print(sample_mismatches)
						
							mismatches.append((slot, cell_id))
				else:
					mismatches.append((slot, cell_id))
					print(f"Mismatch found for Slot {slot}, CellId {cell_id}")
					print(f"FH X_tf[0,2]: {[f'{x:.7f}' for x in fh_x_tf[0,2,0:10]]}")
					print(f"TV X_tf[0,2]: {[f'{x:.7f}' for x in tv_x_tf[0,2,0:10]]}")
					with np.errstate(divide='ignore', invalid='ignore'):
						ratio = fh_x_tf / tv_x_tf
						ratio[ratio == np.inf] = 0
						ratio = np.nan_to_num(ratio)
					print(f"Ratio[0,2]: {[f'{x:.7f}' for x in ratio[0,2,0:10]]}")
					print(f"Ratio range: {np.min(ratio)} to {np.max(ratio)}")
					print(f"Ratio mean: {np.mean(ratio)}")
					print(f"Ratio std: {np.std(ratio)}")
			else:
				print(f"Shape mismatch for Slot {slot}, CellId {cell_id}")
				print(f"FH shape: {fh_x_tf.shape}, TV shape: {tv_x_tf.shape}")
				mismatchCount += 1
				mismatches.append((slot, cell_id))

	# Print summary
	print("\nSummary:")
	print(f"Total matches: {len(matches)}")
	print(f"Total mismatches: {len(mismatches)}")
	print("\nMismatching slots/cells:")
	for slot, cell_id in mismatches:
		print(f"Slot {slot}, CellId {cell_id}")

	if len(mismatches) > 0:
		retCode = 1
	
	return retCode


if __name__ == "__main__":
	import sys
	printMismatches = False
	retCode = 0

	# Set CUBB_HOME  path
	CUBB_HOME = os.environ.get('CUBB_HOME', '/opt/nvidia/cuBB')

	if len(sys.argv) < 2:
		print(f"\nUsage: {sys.argv[0]} <launch_pattern_file.yaml> [printMismatches]\n")
		sys.exit(1)
	if len(sys.argv) == 3:
		printMismatches = sys.argv[2]

	cell_configs, slot_configs = read_yaml_and_h5(CUBB_HOME, sys.argv[1])

	get_clickhouse_stats()

	# Compare slot configs with database
	setup_columns(slot_configs)
	drop_startup_rows(slot_configs)
	retCode = compare_tv_to_fapi(slot_configs, printMismatches)
	retCode = compare_tv_to_fh(slot_configs)
	sys.exit(retCode)

