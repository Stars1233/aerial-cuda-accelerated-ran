/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <sys/socket.h>
#include <sys/un.h>

#include "nv_ipc_efd.h"
#include "nv_ipc_utils.h"

#define TAG (NVLOG_TAG_BASE_NVIPC + 9) //"NVIPC.FD_SHARE"

#define MAXLINE 4096 /* max line length */

/* size of control buffer to send/recv one file descriptor */
#define CONTROLLEN CMSG_LEN(sizeof(int))

/*
 * Pass a file descriptor to another process.
 * If fd<0, then -fd is sent back instead as the error status.
 */
int send_fd(int fd, int fd_to_send)
{
    struct iovec    iov[1];
    struct msghdr   msg;
    char            buf[2]; /* send_fd()/recv_fd() 2-byte protocol */
    struct cmsghdr* cmptr = NULL;

    iov[0].iov_base = buf;
    iov[0].iov_len  = 2;
    msg.msg_iov     = iov;
    msg.msg_iovlen  = 1;
    msg.msg_name    = NULL;
    msg.msg_namelen = 0;

    if(fd_to_send < 0)
    {
        msg.msg_control    = NULL;
        msg.msg_controllen = 0;
        msg.msg_flags      = 0;
        buf[1]             = -fd_to_send; /* nonzero status means error */
        if(buf[1] == 0)
            buf[1] = 1; /* -256, etc. would screw up protocol */
    }
    else
    {
        if(cmptr == NULL && (cmptr = malloc(CONTROLLEN)) == NULL)
            return -1;
        cmptr->cmsg_level       = SOL_SOCKET;
        cmptr->cmsg_type        = SCM_RIGHTS;
        cmptr->cmsg_len         = CONTROLLEN;
        msg.msg_control         = cmptr;
        msg.msg_controllen      = CONTROLLEN;
        msg.msg_flags           = 0;
        *(int*)CMSG_DATA(cmptr) = fd_to_send; /* the fd to pass */
        buf[1]                  = 0;          /* zero status means OK */
    }

    buf[0] = 0; /* null byte flag to recv_fd() */
    if(sendmsg(fd, &msg, 0) != 2)
    {
        free(cmptr);
        return -1;
    }
    free(cmptr);
    return 0;
}

/*
 * Receive a file descriptor from a server process.  Also, any data
 * received is passed to (*userfunc)(STDERR_FILENO, buf, nbytes).
 * We have a 2-byte protocol for receiving the fd from send_fd().
 */
int recv_fd(int fd)
{
    int             newfd, nr, status;
    char*           ptr;
    char            buf[MAXLINE];
    struct iovec    iov[1];
    struct msghdr   msg;
    struct cmsghdr* cmptr = NULL;

    status = -1;
    int retry_count = 0;
    for(retry_count = 0; retry_count < 10; retry_count++)
    {
        iov[0].iov_base = buf;
        iov[0].iov_len  = sizeof(buf);
        msg.msg_iov     = iov;
        msg.msg_iovlen  = 1;
        msg.msg_name    = NULL;
        msg.msg_namelen = 0;
        if(cmptr == NULL && (cmptr = malloc(CONTROLLEN)) == NULL)
            return -1;
        msg.msg_control    = cmptr;
        msg.msg_controllen = CONTROLLEN;
        msg.msg_flags      = 0;
        if((nr = recvmsg(fd, &msg, 0)) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "recvmsg error");
            free(cmptr);
            return -1;
        }
        else if(nr == 0)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "connection closed by server");
            free(cmptr);
            return -1;
        }

        /*
         * See if this is the final data with null & status.  Null
         * is next to last byte of buffer; status byte is last byte.
         * Zero status means there is a file descriptor to receive.
         */
        for(ptr = buf; ptr < &buf[nr];)
        {
            if(*ptr++ == 0)
            {
                if(ptr != &buf[nr - 1])
                    NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "message format error");
                status = *ptr & 0xFF; /* prevent sign extension */
                if(status == 0)
                {
                    if(msg.msg_controllen != CONTROLLEN)
                        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "status = 0 but no fd");
                    newfd = *(int*)CMSG_DATA(cmptr);
                }
                else
                {
                    newfd = -status;
                }
                nr -= 2;
            }
        }
        if(nr > 0)
        {
            NVLOGW(TAG, "%s: nr=%d", __func__, nr);
        }
        if(status >= 0)
        {
            free(cmptr);
            return newfd;
        }
    }

    if (cmptr != NULL)
    {
        free(cmptr);
    }
    NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "recv_fd failed after %d retries", retry_count);
    return -1;
}

void unix_sock_address_init(struct sockaddr_un* addr, char* path)
{
    addr->sun_family = AF_UNIX;
    nvlog_safe_strncpy(addr->sun_path, path, NV_UNIX_SOCKET_PATH_MAX_LEN);
}

int unix_sock_create(struct sockaddr_un* addr)
{
    int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if(sock_fd < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Unix socket create failed");
        return -1;
    }
    socklen_t size = offsetof(struct sockaddr_un, sun_path) + strlen(addr->sun_path);
    unlink(addr->sun_path);
    if(bind(sock_fd, (struct sockaddr*)addr, size) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Unix socket bind failed");
        close(sock_fd);
        return -1;
    }
    NVLOGD(TAG, "Unix socket created: fd=%d, path=%s", sock_fd, addr->sun_path);
    return sock_fd;
}

int unix_sock_listen_and_accept(int listen_fd, struct sockaddr_un* client_addr)
{
    int sock_fd;
    if(listen(listen_fd, 20) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Listen error");
        return -1;
    }
    NVLOGI(TAG, "Wait for connection ...");

    socklen_t addr_len = sizeof(struct sockaddr_un);
    if((sock_fd = accept(listen_fd, (struct sockaddr*)client_addr, &addr_len)) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Accept error");
        return -1;
    }
    else
    {
        NVLOGI(TAG, "Accepted client socket connection: %s", client_addr->sun_path);
        return sock_fd;
    }
}

int unix_sock_connect(int sock_fd, struct sockaddr_un* server_addr)
{
    socklen_t addr_len = offsetof(struct sockaddr_un, sun_path) + strlen(server_addr->sun_path);
    if(connect(sock_fd, (struct sockaddr*)server_addr, addr_len) < 0)
    {
        NVLOGI(TAG, "Wait for server socket: %s", server_addr->sun_path);
        return -1;
    }
    else
    {
        NVLOGI(TAG, "Connected to server socket: %s", server_addr->sun_path);
        return 0;
    }
}
