To run the datalakes unit test, run aerial as normal with datalake_samples set to something reasonable like 10000 and data_core to a free core. 
Launch ru_emulator on using a launch pattern file:
	docker exec -it c_aerial_$USER bash -c "sudo -E build/cuPHY-CP/ru-emulator/ru_emulator/ru_emulator DATALAKE 4C"

Launch testmac with the same launch pattern: 
	docker exec -it c_aerial_$USER bash -c "sudo -E  ./build/cuPHY-CP/testMAC/testMAC/test_mac DATALAKE 4C --channels 0xff


When cuphycontroller prints the right number of this message: 
[CTL.DATA_LAKE] Stopping capture for rnti 1 after reaching configured number of samples (6000).

Run the database verification using the same launch pattern in a pyaerial container: 
	docker exec -it pyaerial_$USER bash -c "./cuPHY-CP/data_lake/tests/compare_db_to_lp.py launch_pattern_DATALAKE_4C.yaml"

