/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */
#include "wraparound_buffer.h"
#include <unistd.h>
#include <sys/mman.h>
#include <map>
#include <mutex>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include "log.h"
#include "assertions.h"

struct {
  std::map<void*, size_t> map;
  std::mutex mtx;
} internal_state;


extern "C" void* malloc_wraparound_buffer(size_t size)
{
  size_t pagesize = getpagesize();
  if (size % pagesize != 0) {
    return NULL;
  }

  int fd = memfd_create("wraparound", MFD_CLOEXEC);
  if (fd == -1) return NULL;
  if (ftruncate(fd, size) != 0) {
    close(fd);
    return NULL;
  }

  void* region = mmap(NULL, size * 2, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (region == MAP_FAILED) {
    close(fd);
    return NULL;
  }

  void* buf = mmap(region, size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, fd, 0);
  if (buf == MAP_FAILED) {
    munmap(region, size * 2);
    close(fd);
    return NULL;
  }

  uint8_t *second_half = (uint8_t *)region + size;
  void* mirror = mmap(second_half, size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, fd, 0);
  if (mirror == MAP_FAILED) {
    munmap(region, size * 2);
    close(fd);
    return NULL;
  }

  close(fd);
  std::lock_guard<std::mutex> lock(internal_state.mtx);
  internal_state.map[buf] = size;
  return buf;
}

extern "C" void free_wraparound_buffer(void* buf) {
  if (buf == nullptr) return;
  std::lock_guard<std::mutex> lock(internal_state.mtx);
  size_t size = internal_state.map[buf];
  if (munmap(buf, size * 2) != 0) {
    LOG_W(UTIL, "munmap failed\n");
  }
}
