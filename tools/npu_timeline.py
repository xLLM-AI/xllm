# Copyright 2016 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Timeline visualization for xLLM using Chrome Trace Format."""

import collections
import copy
import json
import os
import re
import sys
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.logger import logger

class _ChromeTraceFormatter(object):
  """A helper class for generating traces in Chrome Trace Format."""

  def __init__(self, show_memory: bool = False) -> None:
    """Constructs a new Chrome Trace formatter."""
    self._show_memory = show_memory
    self._events = []
    self._metadata = []

  def _create_event(
      self,
      ph: str,
      category: str,
      name: str,
      pid: int,
      tid: int,
      timestamp: int,
  ) -> Dict[str, Union[str, int]]:
    """Creates a new Chrome Trace event.

    For details of the file format, see:
    https://github.com/catapult-project/catapult/blob/master/tracing/README.md

    Args:
      ph:  The type of event - usually a single character.
      category: The event category as a string.
      name:  The event name as a string.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      timestamp:  The timestamp of this event as a long integer.

    Returns:
      A JSON compatible event object.
    """
    event = {}
    event['ph'] = ph
    event['cat'] = category
    event['name'] = name
    event['pid'] = pid
    event['tid'] = tid
    event['ts'] = timestamp
    return event

  def emit_pid(self, name: str, pid: int) -> None:
    """Adds a process metadata event to the trace.

    Args:
      name:  The process name as a string.
      pid:  Identifier of the process as an integer.
    """
    event = {}
    event['name'] = 'process_name'
    event['ph'] = 'M'
    event['pid'] = pid
    event['args'] = {'name': name}
    self._metadata.append(event)

  def emit_tid(self, name, pid, tid):
    """Adds a thread metadata event to the trace.

    Args:
      name:  The thread name as a string.
      pid:  Identifier of the process as an integer.
      tid:  Identifier of the thread as an integer.
    """
    event = {}
    event['name'] = 'thread_name'
    event['ph'] = 'M'
    event['pid'] = pid
    event['tid'] = tid
    event['args'] = {'name': name}
    self._metadata.append(event)

  def emit_region(
      self,
      timestamp: int,
      duration: int,
      pid: int,
      tid: int,
      category: str,
      name: str,
      args: Dict[str, Any],
  ) -> None:
    """Adds a region event to the trace.

    Args:
      timestamp:  The start timestamp of this region as a long integer.
      duration:  The duration of this region as a long integer.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      category: The event category as a string.
      name:  The event name as a string.
      args:  A JSON-compatible dictionary of event arguments.
    """
    event = self._create_event('X', category, name, pid, tid, timestamp)
    event['dur'] = duration
    event['args'] = args
    self._events.append(event)

  def emit_obj_create(
      self,
      category: str,
      name: str,
      timestamp: int,
      pid: int,
      tid: int,
      object_id: int,
  ) -> None:
    """Adds an object creation event to the trace.

    Args:
      category: The event category as a string.
      name:  The event name as a string.
      timestamp:  The timestamp of this event as a long integer.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      object_id: Identifier of the object as an integer.
    """
    event = self._create_event('N', category, name, pid, tid, timestamp)
    event['id'] = object_id
    self._events.append(event)

  def emit_obj_delete(
      self,
      category: str,
      name: str,
      timestamp: int,
      pid: int,
      tid: int,
      object_id: int,
  ) -> None:
    """Adds an object deletion event to the trace.

    Args:
      category: The event category as a string.
      name:  The event name as a string.
      timestamp:  The timestamp of this event as a long integer.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      object_id: Identifier of the object as an integer.
    """
    event = self._create_event('D', category, name, pid, tid, timestamp)
    event['id'] = object_id
    self._events.append(event)

  def emit_obj_snapshot(
      self,
      category: str,
      name: str,
      timestamp: int,
      pid: int,
      tid: int,
      object_id: int,
      snapshot: Dict[str, Any],
  ) -> None:
    """Adds an object snapshot event to the trace.

    Args:
      category: The event category as a string.
      name:  The event name as a string.
      timestamp:  The timestamp of this event as a long integer.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      object_id: Identifier of the object as an integer.
      snapshot:  A JSON-compatible representation of the object.
    """
    event = self._create_event('O', category, name, pid, tid, timestamp)
    event['id'] = object_id
    event['args'] = {'snapshot': snapshot}
    self._events.append(event)

  def emit_flow_start(
      self, name: str, timestamp: int, pid: int, tid: int, flow_id: int
  ) -> None:
    """Adds a flow start event to the trace.

    When matched with a flow end event (with the same 'flow_id') this will
    cause the trace viewer to draw an arrow between the start and end events.

    Args:
      name:  The event name as a string.
      timestamp:  The timestamp of this event as a long integer.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      flow_id: Identifier of the flow as an integer.
    """
    event = self._create_event('s', 'DataFlow', name, pid, tid, timestamp)
    event['id'] = flow_id
    self._events.append(event)

  def emit_flow_end(
      self, name: str, timestamp: int, pid: int, tid: int, flow_id: int
  ) -> None:
    """Adds a flow end event to the trace.

    When matched with a flow start event (with the same 'flow_id') this will
    cause the trace viewer to draw an arrow between the start and end events.

    Args:
      name:  The event name as a string.
      timestamp:  The timestamp of this event as a long integer.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      flow_id: Identifier of the flow as an integer.
    """
    event = self._create_event('t', 'DataFlow', name, pid, tid, timestamp)
    event['id'] = flow_id
    self._events.append(event)

  def emit_counter(
      self,
      category: str,
      name: str,
      pid: int,
      timestamp: int,
      counter: str,
      value: int,
  ) -> None:
    """Emits a record for a single counter.

    Args:
      category: The event category as a string.
      name:  The event name as a string.
      pid:  Identifier of the process generating this event as an integer.
      timestamp:  The timestamp of this event as a long integer.
      counter: Name of the counter as a string.
      value:  Value of the counter as an integer.
    """
    event = self._create_event('C', category, name, pid, 0, timestamp)
    event['args'] = {counter: value}
    self._events.append(event)

  def emit_counters(self, category, name, pid, timestamp, counters):
    """Emits a counter record for the dictionary 'counters'.

    Args:
      category: The event category as a string.
      name:  The event name as a string.
      pid:  Identifier of the process generating this event as an integer.
      timestamp:  The timestamp of this event as a long integer.
      counters: Dictionary of counter values.
    """
    event = self._create_event('C', category, name, pid, 0, timestamp)
    event['args'] = counters.copy()
    self._events.append(event)

  def format_to_string(self, pretty: bool = False) -> str:
    """Formats the chrome trace to a string.

    Args:
      pretty: (Optional.)  If True, produce human-readable JSON output.

    Returns:
      A JSON-formatted string in Chrome Trace format.
    """
    trace = {}
    trace['traceEvents'] = self._metadata + self._events
    if pretty:
      return json.dumps(trace, indent=4, separators=(',', ': '))
    else:
      return json.dumps(trace, separators=(',', ':'))


class Timeline(object):
  """A class for visualizing execution timelines of xLLM steps."""

  def __init__(self, log_file_path: str) -> None:
    """Constructs a new Timeline.

    A 'Timeline' is used for visualizing the execution of a xLLM
    computation.  It shows the timings and concurrency of execution at
    the granularity of xLLM Ops.
    This class is not thread safe.
    """

    self._step_stats = self.parse_log(log_file_path)
    self._chrome_trace = _ChromeTraceFormatter()
    self._next_pid = 0
    self._marker_names = {}  # id -> trace name for marker.
    self._marker_end_ts = {} # id -> (deviceId, end timestamp) for marker.
    self._device_pids = {}  # device id -> trace pid for marker.
    self._memory_pids = {}  # device id -> trace pid for memory.
    self._kernel_pids = {}  # device id -> trace pid for kernel.
    self._next_flow_id = 0
    self._flow_starts = {}  # tensor_name -> (timestamp, pid, tid)

  def parse_log(self, log_file_path: str) -> List:
    import json
    step_stats = []
    with open(log_file_path, 'r') as f:
      lines = f.readlines()
      for line in lines:
        if "AscendKind" in line:
          start_idx = line.find('{')
          line = line[start_idx:].strip()
          stats = json.loads(line)
          if stats["AscendKind"] in ['MARKER', 'MEMORY', 'KERNEL']:
            step_stats.append(stats)
    assert len(step_stats) > 0, "step_stats is empty"
    return step_stats

  def _alloc_pid(self) -> int:
    """Allocate a process Id."""
    pid = self._next_pid
    self._next_pid += 1
    return pid

  def _alloc_flow_id(self) -> int:
    """Allocate a flow Id."""
    flow_id = self._next_flow_id
    self._next_flow_id += 1
    return flow_id

  def _emit_marker(
      self, stats: Dict, pid: int
  ) -> None:
    """Generates a Chrome Trace event to show marker event.

    Args:
      stats: The log recording marker event.
      pid: The pid assigned for the device where this marker stat ran.
    """
    name = stats['name']
    start = stats['timestamp'] / 1000 #microsecond 
    duration = stats['duration'] / 1000 #microsecond 
    tid = stats['streamId']
    sourceKind = stats['sourceKind']
    flag = stats['flag']
    args = {'sourceKind': sourceKind, 'flag': flag}
    self._chrome_trace.emit_region(start, duration, pid, tid, 'Marker', name, args)

  def _emit_memory(
      self, stats: Dict, pid: int
  ) -> None:
    """Generates a Chrome Trace event to show memory event.

    Args:
      stats: The log recording memory event.
      pid: The pid assigned for the device where this memory stat ran.
    """
    name = "memory_alloc" if 1 == stats['memoryKind'] else "memory_free"
    start = stats['start'] / 1000 #microsecond 
    duration = stats['duration'] / 1000 #microsecond 
    tid = stats['streamId']
    address = stats['address']
    bytes_ = stats['bytes'] / 1024 / 1024
    args = {'address': address, 'bytes': bytes_}
    self._chrome_trace.emit_region(start, duration, pid, tid, 'Memory', name, args)

  def _emit_kernel(
      self, stats: Dict, pid: int
  ) -> None:
    """Generates a Chrome Trace event to show kernel event.

    Args:
      stats: The log recording kernel event.
      pid: The pid assigned for the device where this kernel stat ran.
    """
    name = stats['name'] if stats['name'] != "" else stats['type']
    start = stats['start'] / 1000 #microsecond 
    duration = stats['duration'] / 1000 #microsecond 
    tid = stats['streamId']
    type_ = stats['type']
    args = {'type': type_}
    self._chrome_trace.emit_region(start, duration, pid, tid, 'Kernel', name, args)  

  def _allocate_pids(self) -> None:
    """Allocate fake process ids for each device in the step_stats_pb2.StepStats."""
    # Add processes in the Chrome trace to show compute and data activity.
    for dev_stats in self._step_stats:
      deviceId = dev_stats['deviceId']
      if dev_stats['AscendKind'] == "MARKER":
        if dev_stats['name'] != "":
          self._marker_names[dev_stats['id']] = dev_stats['name']
        else:
          if dev_stats['flag'] == 32 or dev_stats['flag'] == 4: # mstxRangeEnd
            if dev_stats['id'] not in self._marker_end_ts:
              self._marker_end_ts[dev_stats['id']] = [(dev_stats['deviceId'], dev_stats['timestamp'])]
            else:
              self._marker_end_ts[dev_stats['id']].append((dev_stats['deviceId'], dev_stats['timestamp']))
        if deviceId not in self._device_pids:
          device_pid = self._alloc_pid()
          self._device_pids[deviceId] = device_pid
          if deviceId < 50:
            self._chrome_trace.emit_pid('CPU Process ' + str(deviceId), device_pid)
          else:
            self._chrome_trace.emit_pid('NPU Device ' + str(deviceId), device_pid)
      elif dev_stats['AscendKind'] == "MEMORY":
        if deviceId not in self._memory_pids:
          device_pid = self._alloc_pid()
          self._memory_pids[deviceId] = device_pid
          self._chrome_trace.emit_pid('Memory ' + str(deviceId), device_pid)
      elif dev_stats['AscendKind'] == "KERNEL":
        if deviceId not in self._kernel_pids:
          device_pid = self._alloc_pid()
          self._kernel_pids[deviceId] = device_pid
          self._chrome_trace.emit_pid('Kernel ' + str(deviceId), device_pid)
      else:
        logger.warning(f"Unsupport AscendKind {dev_stats['AscendKind']}")

  def _get_marker_end(self, deviceId:int, stat_id:int) -> Dict:
    """Get the end marker stats."""
    for dev_stats in self._step_stats:
      if 'MARKER' not in dev_stats['AscendKind']:
        continue
      if dev_stats['flag'] == 32 or dev_stats['flag'] == 4: # mstxRangeEnd
        cur_deviceId = dev_stats['deviceId']
        cur_id = dev_stats['id']
        if cur_deviceId == deviceId and cur_id == stat_id:
          return dev_stats
    return None

  def _show_marker(self, show_flow: bool = False) -> None:
    """Visualize the marker activity."""
    for dev_stats in self._step_stats:
      if 'MARKER' in dev_stats['AscendKind']:
        deviceId = dev_stats['deviceId']
        device_pid = self._device_pids[deviceId]
        if dev_stats['flag'] == 32 or dev_stats['flag'] == 4: # mstxRangeEnd
          continue
        start_time = dev_stats['timestamp']
        stats_id = dev_stats['id']
        end_time = 0
        # end_marker_stat = self._get_marker_end(deviceId, stats_id)
        for end_ts in self._marker_end_ts[stats_id]:
          cur_deviceId, cur_end_time = end_ts
          if cur_deviceId == deviceId:
            end_time = cur_end_time
        if end_time == 0:
          logger.warning(f"end marker not found: deviceId:{deviceId} id:{stats_id}")
          continue
        # end_time = end_marker_stat['timestamp']
        dev_stats['duration'] = end_time - start_time
        dev_stats['name'] = self._marker_names[stats_id]
        self._emit_marker(dev_stats, device_pid)

  def _show_memory(self) -> None:
    """Visualize the memory activity."""
    for dev_stats in self._step_stats:
      if 'MEMORY' in dev_stats['AscendKind']:
        deviceId = dev_stats['deviceId']
        device_pid = self._memory_pids[deviceId]
        start_time = dev_stats['start']
        end_time = dev_stats['end']
        dev_stats['duration'] = end_time - start_time
        self._emit_memory(dev_stats, device_pid)

  def _show_kernel(self) -> None:
    """Visualize the kernel activity."""
    for dev_stats in self._step_stats:
      if 'KERNEL' in dev_stats['AscendKind']:
        deviceId = dev_stats['deviceId']
        device_pid = self._kernel_pids[deviceId]
        start_time = dev_stats['start']
        end_time = dev_stats['end']
        dev_stats['duration'] = end_time - start_time
        self._emit_kernel(dev_stats, device_pid)
      
  def generate_chrome_trace_format(
      self,
  ) -> str:
    # pyformat: disable
    """Produces a trace in Chrome Trace Format.
    Returns:
      A JSON formatted string in Chrome Trace format.
    """
    # pyformat: enable
    self._allocate_pids()
    self._show_marker()
    self._show_memory()
    self._show_kernel()
    return self._chrome_trace.format_to_string(pretty=True)

def parse_args():
  parser = argparse.ArgumentParser(description='Generate Chrome trace from log file')
  parser.add_argument('--input', '-i', type=str, default='./node_0.log',
                      help='Path to input log file (default: ./log/node_0.log)')
  parser.add_argument('--output', '-o', type=str, default='mspti_chrome_trace.json',
                      help='Path to output Chrome trace file (default: mspti_chrome_trace.json)')
  return parser.parse_args()

# main
if  __name__ == '__main__':
  args = parse_args()
  time_line = Timeline(args.input)
  chrome_trace_str = time_line.generate_chrome_trace_format()
  with open(args.output, 'w') as f:
      f.write(chrome_trace_str)
